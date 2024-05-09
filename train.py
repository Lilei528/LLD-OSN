import json
import os
import pathlib
import time
import datetime
import argparse

import torch
import torch.nn.functional as F

import bert_model
from transformer_model import TransformerModel

from apex import amp

import dataloader_text
import transformer_model
import utils.core
from utils.builder import build_sgd_optimizer, build_adam_optimizer
from utils.category_vector_data import category_datas
from utils.core import accuracy, evaluate
from utils.mahalanobis_torch import m_dist
from utils.make_data import load_dataset

from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, step_flagging
from utils.plotter import plot_results_cotraining
from utils.loss import cross_entropy, entropy_loss, regression_loss, BCEFocalLoss
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
'''
load_dataset 数据集进行了切片
transformer层一般为12层
优化器的选择
acc函数没写
'''

class CLDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def flat_label(label, prob, delta):
    """
    Calculates the modified probabilities for each label based on the given inputs.

    Args:
        label (torch.Tensor): Batch of label probabilities (shape: batch_size x label_len).
        prob (torch.Tensor): Confidence scores for each label (shape: batch_size).
        delta (float): Hyperparameter.

    Returns:
        torch.Tensor: Modified probabilities for each label (shape: batch_size x label_len).
    """
    # Calculate the exponentiated term for each label
    exp_term = torch.exp((1 - prob.unsqueeze(1)) * label / delta)

    # Normalize each row (across labels) by dividing by the sum of probabilities for that row
    row_sums = exp_term.sum(dim=1, keepdim=True)
    normalized_probs = exp_term / row_sums

    return normalized_probs


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_lr_plan(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs

    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))

    return lrs

def update_mahalanobis_datas(clean_datas,clean_labels,mahalanobis_datas:category_datas):
    '''
    :param clean_datas:干净的样本
    clean_labels: 对应标签
    :param mahalanobis_datas: 待更新类别信息的类
    :return:
    '''
    mahalanobis_datas.add_some_datas(clean_datas.cpu(),clean_labels.cpu())

def get_ood_samples_index(noiy_data,mahalanobis_datas:category_datas,index,threshold):
    ood_index=[]

    for i in range(noiy_data.size(0)):
        one_sent=noiy_data[i]
        one_list=[]
        for j in range(mahalanobis_datas.label_len):
            d = m_dist(mahalanobis_datas.mean[j], one_sent, mahalanobis_datas.cov_rev[j])
            one_list.append(d.item())

        if min(one_list)>threshold:
            ood_index.append(index[i])

    return torch.LongTensor(ood_index) # (noisy_size,class_num)



def get_smoothed_label_distribution(labels, nc, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), nc), fill_value=epsilon / (nc - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def sample_selector(l1, l2, drop_rate):
    ind_sorted_1 = torch.argsort(l1.data)  # ascending order
    ind_sorted_2 = torch.argsort(l2.data)
    num_remember = max(int((1 - drop_rate) * l1.shape[0]), 1)
    ind_clean_1 = ind_sorted_1[:num_remember]
    ind_clean_2 = ind_sorted_2[:num_remember]
    ind_unclean_1 = ind_sorted_1[num_remember:]
    ind_unclean_2 = ind_sorted_2[num_remember:]
    return {'clean1': ind_clean_1, 'clean2': ind_clean_2, 'unclean1': ind_unclean_1, 'unclean2': ind_unclean_2}

from tqdm import tqdm
import warnings

def main(cfg, device):
    warnings.filterwarnings("ignore")

    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, f'{logtime}-{cfg.log}')
    # result_dir = os.path.join(logger_root, f'ablation_study-{cfg.log}')  #TODO
    logger = Logger(logging_dir=result_dir, DEBUG=False)
    logger.set_logfile(logfile_name='log.txt')
    # 保存输入参数
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    # 保存运行输出内容到 log.txt
    logger.debug(f'Result Path: {result_dir}')

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    n_classes = cfg.n_classes

    net1 = bert_model.bone_BERT(n_classes,pretrained_model_name=cfg.pretrainedmodel)
    net2 = bert_model.bone_BERT(n_classes,pretrained_model_name=cfg.pretrainedmodel)

    optimizer1 = build_sgd_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay,net=net1)
    optimizer2 = build_sgd_optimizer(net2.parameters(), cfg.lr, cfg.weight_decay,net=net2)

    net1=net1.to(device)
    net2=net2.to(device)


    net1,optimizer1 = amp.initialize(net1, optimizer1, opt_level=opt_lvl, keep_batchnorm_fp32=None,
                                      loss_scale=None, verbosity=0)
    net2 ,optimizer2= amp.initialize(net2, optimizer2, opt_level=opt_lvl, keep_batchnorm_fp32=None,
                                      loss_scale=None, verbosity=0)

    # 学习率变化
    lr_plan = make_lr_plan(cfg.lr, cfg.stage1, cfg.epochs)

    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net1.__repr__())
        f.write('\n\n---------------------------\n\n')
        f.writelines(net2.__repr__())

    # drop rate scheduler ----------------------------------------------------------------------------------------------------------------------------
    T_k = cfg.stage1
    final_drop_rate = 0.25
    final_ldl_rate = cfg.ldl_rate
    drop_rate_scheduler = np.ones(cfg.epochs) * final_drop_rate
    drop_rate_scheduler[:T_k] = np.linspace(0, final_drop_rate, T_k)
    drop_rate_scheduler[T_k:cfg.epochs] = np.linspace(final_drop_rate, final_ldl_rate, cfg.epochs - T_k)

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    train_data = load_dataset(data_dir,batch_size=cfg.batch_size,pretrainedmodel=cfg.pretrainedmodel,max_len=max_len,model='train')
    test_data = load_dataset(test_data_dir, batch_size=cfg.batch_size,pretrainedmodel=cfg.pretrainedmodel,max_len=max_len,model='test')


    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss1, train_loss2 = AverageMeter(), AverageMeter()
    train_f1, train_f2 = get_f1(), get_f1()
    iter_time = AverageMeter()

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_f1, best_f2 = 0.0, 0.0
    best_epoch1, best_epoch2 = 0, 0

    net1_mahalanobis_datas = category_datas(n_class=n_classes, category_keep_num= 800)

    net2_mahalanobis_datas = category_datas(n_class=n_classes, category_keep_num= 800)

    t = torch.tensor(train_data.label_list)
    labels2learn1 = torch.full(size=(len(train_data.label_list), n_classes), fill_value=0.0)
    labels2learn1.scatter_(dim=1, index=torch.unsqueeze(t, dim=1), value=1.0 * 10)
    labels2learn2 = labels2learn1

    flag = [0, 0, 0]
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        train_loss1.reset()
        train_f1.reset()
        train_loss2.reset()
        train_f2.reset()

        net1.train()
        net2.train()
        #adjust_lr(optimizer1, lr_plan[epoch])
        #adjust_lr(optimizer2, lr_plan[epoch])
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        print(f'开始 epoch {epoch} : 开始训练  ')
        with tqdm(total=train_data.total_batch) as pbar:
            # train this epoch
            for it, sample in enumerate(train_data.get_batch_data()):
                pbar.update(1)
                s = time.time()

                # optimizer1.zero_grad()
                # optimizer2.zero_grad()

                indices = sample['index_list']
                x1 = sample['x'].to(device)
                y0 = torch.tensor(sample['y']).to(device)
                y = get_smoothed_label_distribution(y0, nc=n_classes, epsilon=cfg.smooth_rate)
                if (x1.size(0)==0):
                    continue

                padding_mask = torch.tensor(sample['mask'])
                padding_mask = padding_mask.to(0)

                output1 = net1(x1, attention_mask = padding_mask)
                output2 = net2(x1, attention_mask = padding_mask)
                logits1 = output1['logits']
                logits2 = output2['logits']
                prob1=output1['prob']
                prob2=output2['prob']
                if epoch < cfg.stage1:  # warmup
                    if flag[0] == 0:
                        step_flagging('stage 1')
                        flag[0] += 1

                    loss1 = cross_entropy(prob1, y)
                    loss2 = cross_entropy(prob2, y)
                    with torch.no_grad():
                        cce_losses1 = cross_entropy(prob1, y, reduction='none')
                        cce_losses2 = cross_entropy(prob2, y, reduction='none')

                        sample_selection = sample_selector(cce_losses1, cce_losses2, drop_rate_scheduler[epoch])
                        update_mahalanobis_datas(logits1[sample_selection['clean1']], y0[sample_selection['clean1']],
                                                 net1_mahalanobis_datas)
                        update_mahalanobis_datas(logits2[sample_selection['clean2']], y0[sample_selection['clean2']],
                                                 net2_mahalanobis_datas)
                else:  # learn label distributions
                    if flag[1] == 0:
                        step_flagging('stage 2')
                        flag[1] += 1
                    if x1.size(0)==1:
                        break
                    with torch.no_grad():
                        # 对于下标中的后缀， 后缀数字表示模型k认为的数据类型
                        cce_losses1 = cross_entropy(output1['prob'], y, reduction='none')
                        cce_losses2 = cross_entropy(output2['prob'], y, reduction='none')
                        losses1 = cce_losses1
                        losses2 = cce_losses2
                        sample_selection = sample_selector(losses1, losses2, drop_rate_scheduler[epoch])
                        # net1_mahalanobis_datas  维护了net2认为干净的在net1中的历史均值和方差
                        update_mahalanobis_datas(logits1[sample_selection['clean2']], y0[sample_selection['clean2']],
                                                 net1_mahalanobis_datas
                                                 )
                        update_mahalanobis_datas(logits2[sample_selection['clean1']], y0[sample_selection['clean1']],
                                                 net2_mahalanobis_datas
                                                     )
                        # 使用net1维护的历史均值和方差对net1认为的noisy数据进行马氏距离判断 ，net1认为的ood金标
                        maha_ood_index_1 = get_ood_samples_index(logits1[sample_selection['unclean1']].cpu(),
                                                            net1_mahalanobis_datas,
                                                            sample_selection['unclean1'], cfg.threshold
                                                            )
                        maha_ood_index_2 = get_ood_samples_index(logits2[sample_selection['unclean2']].cpu(),
                                                            net2_mahalanobis_datas,
                                                            sample_selection['unclean2'], cfg.threshold
                                                            )
                        maha_ood_index_1 =maha_ood_index_1.to(0)
                        maha_ood_index_2 =maha_ood_index_2.to(0)
                        # net1分类头认为的ood数据下标，用于输入net2
                        pred_ood_index1=(output1['ood_prob'].squeeze(dim=1)>0.5).nonzero().view(-1)
                        pred_ood_index2=(output2['ood_prob'].squeeze(dim=1)>0.5).nonzero().view(-1)
                        pred_ood_index1 = torch.LongTensor([index.item() for index in pred_ood_index1 if index in sample_selection['unclean1']]).to(0)
                        pred_ood_index2 = torch.LongTensor([index.item() for index in pred_ood_index2 if index in sample_selection['unclean2']]).to(0)
                        # net1分类头预测的（net2认为的ood数据）的ood概率，用于net1的扁平化
                        prob_noisy_ood1 =torch.tensor([prob.item() for prob in output1['ood_prob'][pred_ood_index2].detach() ]).to(0)
                        prob_noisy_ood2 =torch.tensor([prob.item() for prob in output2['ood_prob'][pred_ood_index1].detach()  ]).to(0)

                        # net1 认为的id数据下标，用于输入net2
                        mask = ~sample_selection['unclean1'].unsqueeze(1).eq(pred_ood_index1).any(1)
                        id_index_1 = sample_selection['unclean1'][mask]
                        mask = ~sample_selection['unclean2'].unsqueeze(1).eq(pred_ood_index2).any(1)
                        id_index_2 = sample_selection['unclean2'][mask]

                    #  获得所有的样本的软标签
                    y_t1 = labels2learn1[indices, :].clone().to(device)
                    y_t2 = labels2learn2[indices, :].clone().to(device)
                    y_t1.requires_grad = True
                    y_t2.requires_grad = True
                    y_d1 = F.softmax(y_t1, dim=1) + 1e-8
                    y_d2 = F.softmax(y_t2, dim=1) + 1e-8

                    # 干净样本的损失
                    prob_clean1 = prob1[sample_selection['clean2']]
                    prob_clean2 = prob2[sample_selection['clean1']]
                    y_clean1 = y[sample_selection['clean2']]
                    y_clean2 = y[sample_selection['clean1']]
                    losses_clean1 = cross_entropy(prob_clean1, y_clean1, reduction='none')   # (Nc1)
                    losses_clean2 = cross_entropy(prob_clean2, y_clean2, reduction='none')   # (Nc2)
                    first_loss_clean1 = losses_clean1.mean()
                    first_loss_clean2 = losses_clean2.mean()

                    sec_loss_clean1 = cross_entropy(y_t1[sample_selection['clean2']], y[sample_selection['clean2']])
                    sec_loss_clean2 = cross_entropy(y_t2[sample_selection['clean1']], y[sample_selection['clean1']])

                    # id噪声的损失
                    # 获得id数据软标签
                    if id_index_1.size(0)!=0:
                        prob_id2 = prob2[id_index_1]
                        y_d_id2 = y_d2[id_index_1]
                        w2 = np.random.beta(cfg.phi, cfg.phi, prob_id2.size(0))
                        w2 = x1.new(w2).view(prob_id2.size(0), 1)
                        # idx2為選擇的乾净樣本
                        idx2 = np.random.choice(sample_selection['clean1'].cpu().numpy(), y_d_id2.size(0),
                                                replace=False if sample_selection['clean1'].size(0) >= y_d_id2.size(
                                                    0) else True)
                        idx2 = torch.tensor(idx2).to(device)



                        mixed_x2 = w2 * logits2[id_index_1] + (1 - w2) * logits2[idx2]
                        mixed_y2 = w2 * y_d_id2 + (1 - w2) * y_d2[idx2]


                        mixed_prob2 = net2.head_get_prob(mixed_x2)

                        loss_id2 = kl_div(F.softmax(mixed_prob2, dim=1), mixed_y2).mean()
                    if id_index_2.size(0)!=0:
                        prob_id1 = prob1[id_index_2]
                        y_d_id1 = y_d1[id_index_2]
                        # id clean混合的权重
                        w1 = np.random.beta(cfg.phi, cfg.phi, prob_id1.size(0))
                        w1 = x1.new(w1).view(prob_id1.size(0), 1)
                        # 随机选择clean样本
                        idx1 = np.random.choice(sample_selection['clean2'].cpu().numpy(), prob_id1.size(0),
                                                replace=False if sample_selection['clean2'].size(0) >= prob_id1.size(
                                                    0) else True)
                        idx1 = torch.tensor(idx1).to(device)

                        # 混合id和clean样本,id_index_2 为id的index，idx1为干净的下标
                        mixed_x1 = w1 * logits1[id_index_2] + (1 - w1) * logits1[idx1]
                        mixed_y1 = w1 * y_d_id1 + (1 - w1) * y_d1[idx1]
                        # 混合样本输入模型
                        mixed_prob1 = net1.head_get_prob(mixed_x1)

                        # KL散度用与mixup
                        loss_id1 = kl_div(F.softmax(mixed_prob1, dim=1) , mixed_y1).mean()

                    # 分布外损失
                    # ood分类头训练
                    ood_y1=torch.tensor([0 for  i in range(y0.size(0)) ])
                    ood_y2=torch.tensor([0 for  i in range(y0.size(0)) ])
                    ood_y1[maha_ood_index_2]=1
                    ood_y2[maha_ood_index_1]=1
                    prob_ood1=output1['ood_prob']
                    prob_ood2=output2['ood_prob']
                    first_loss_ood1=BCEFocalLoss(prob_ood1, ood_y1.to(0))
                    first_loss_ood2=BCEFocalLoss(prob_ood2, ood_y2.to(0))
                    # 类别分类头扁平化

                    if pred_ood_index1.size(0)!=0:
                        prob_class_ood2 = output2['prob'][pred_ood_index1]
                        ood_y_d2 = y_d2[pred_ood_index1]
                        ood_flat_d2= flat_label( ood_y_d2,prob_noisy_ood2,cfg.delta )
                        sec_loss_ood2 = kl_div(ood_flat_d2, F.softmax(prob_class_ood2)).mean()

                    if pred_ood_index2.size(0) != 0:
                        prob_class_ood1 = output1['prob'][pred_ood_index2]
                        ood_y_d1 = y_d1[pred_ood_index2]
                        ood_flat_d1= flat_label( ood_y_d1,prob_noisy_ood1,cfg.delta )
                        sec_loss_ood1 = kl_div(ood_flat_d1, F.softmax(prob_class_ood1)).mean()

                    if id_index_1.size(0)==0:
                        loss2 = cfg.alpha * sec_loss_clean2 + (1 - cfg.alpha) * (first_loss_clean2 + cfg.beta * sec_loss_ood2)+ cfg.gamma * first_loss_ood2

                    elif pred_ood_index1.size(0)==0:
                         loss2 = cfg.alpha * sec_loss_clean2 + (1 - cfg.alpha) * (first_loss_clean2 + cfg.beta * loss_id2)+cfg.gamma * first_loss_ood2
                    else:
                        loss2 = cfg.alpha*sec_loss_clean2 +(1-cfg.alpha)*(first_loss_clean2 +cfg.beta*sec_loss_ood2+  cfg.beta*loss_id2    )+ cfg.gamma * first_loss_ood2

                    if id_index_2.size(0) == 0:
                        loss1 = cfg.alpha*sec_loss_clean1 +(1-cfg.alpha)*(first_loss_clean1 +cfg.beta*sec_loss_ood1 )\
                                + cfg.gamma * first_loss_ood1

                    elif pred_ood_index2.size(0) == 0:
                        loss1 = cfg.alpha * sec_loss_clean1 + (1 - cfg.alpha) * ( first_loss_clean1  + cfg.beta * loss_id1)\
                                + cfg.gamma * first_loss_ood1
                    else:
                        loss1 = cfg.alpha*sec_loss_clean1 +(1-cfg.alpha)*(first_loss_clean1 +cfg.beta*sec_loss_ood1 +  cfg.beta*loss_id1)\
                                + cfg.gamma * first_loss_ood1


                p1= torch.argmax(F.softmax(prob1.detach()),dim=1).tolist()
                p2= torch.argmax(F.softmax(prob2.detach()),dim=1).tolist()

                train_f1.add_some_result(y0.tolist(), p1)
                train_f2.add_some_result(y0.tolist(), p2)

                train_now_f1 = train_f1.cal_f1_score()
                train_now_f2 = train_f2.cal_f1_score()


                train_loss1.update(loss1.item(), x1.size(0))
                train_loss2.update(loss2.item(), x1.size(0))



                if cfg.use_fp16:
                    with amp.scale_loss(loss1, optimizer1) as scaled_loss1:
                        scaled_loss1.backward()
                    with amp.scale_loss(loss2, optimizer2) as scaled_loss2:
                        scaled_loss2.backward()
                else:
                    loss1.backward()
                    loss2.backward()

                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                if epoch >= cfg.stage1:
                   
                    y_t1.data.sub_(cfg.lmd * y_t1.grad.data)
                    y_t2.data.sub_(cfg.lmd * y_t2.grad.data)
                    labels2learn1[indices, :] = y_t1.detach().clone().cpu().data
                    labels2learn2[indices, :] = y_t2.detach().clone().cpu().data
                    del y_t1, y_t2

                iter_time.update(time.time() - s, 1)

                if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_data.label_list)):
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                    mem = torch.cuda.memory_reserved() / 2 ** 30
                    console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                      f"Iter:[{it + 1:>4d}/{len(train_data.label_list):>4d}]  " \
                                      f"Train Accuracy 1:[{train_now_f1['avg']:6.2f}]  " \
                                      f"Train Accuracy 2:[{train_now_f2['avg']:6.2f}]  " \
                                      f"Loss 1:[{train_loss1.avg:4.4f}]  " \
                                      f"Loss 2:[{train_loss2.avg:4.4f}]  " \
                                      f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                      f"{iter_time.avg:6.2f} sec/iter"
                    logger.debug(console_content)

        print(f' epoch {epoch} : 开始测试  ')

        # evaluate this epoch
        test_f1,sss1 = evaluate(test_data, net1, device,"net1")
        test_f2,sss2 = evaluate(test_data, net2, device,"net2")



        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch1 = epoch + 1
            f=open('./pred1true.txt','w',encoding='utf-8')
            f.write(str(best_f1))
            f.close()
            torch.save(net1.state_dict(), f'{result_dir}/net1_best_epoch.pth')
        if test_f2 > best_f2:
            best_f2 = test_f2
            best_epoch2 = epoch + 1
            f=open('./pred2true .txt','w',encoding='utf-8')
            f.write(str(best_f2))
            f.close()
            torch.save(net2.state_dict(), f'{result_dir}/net2_best_epoch.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss(1/2): ({train_loss1.avg:>6.4f}/{train_loss2.avg:>6.4f}) | '
                    f'train accuracy(1/2): ({train_now_f1["avg"]:>6.3f}/{train_now_f2["avg"]:>6.3f}) | '
                    f'test accuracy(1/2): ({test_f1:>6.3f}/{test_f2:>6.3f}) | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy(1/2): ({best_f1:6.3f}/{best_f2:6.3f}) @ epoch: ({best_epoch1:03d}/{best_epoch2:03d})')
        plot_results_cotraining(result_file=f'{result_dir}/log.txt')

    torch.save(labels2learn1, f'{result_dir}/labels_learned.pt')

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    best_accuracy = max(best_f1, best_f2)
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_prefix', type=str, default='co_lnl_ood')
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=200.00)
    parser.add_argument('--net1', type=str, default='resnet18')
    parser.add_argument('--net2', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--stage1', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lmd', type=float, default=200)
    parser.add_argument('--ldl_rate', type=float, default=0.1)
    parser.add_argument('--phi', type=float, default=0.4)
    parser.add_argument('--smooth_rate', type=float, default=0.15)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.35)

    args = parser.parse_args()
    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    config.log = f'{config.net1}-{config.net2}-{config.log_prefix}'

    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(
        f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime / 3600:.3f} hours)')

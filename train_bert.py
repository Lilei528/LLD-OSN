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
builder 分ood
下面 make lr plan是常数
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
        if param_group['mname']=='fc_decay':
            param_group['lr'] = lr


def make_lr_plan(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs

    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))

    return lrs


def update_mahalanobis_datas(clean_datas, clean_labels, mahalanobis_datas: category_datas):
    '''
    :param clean_datas:干净的样本
    clean_labels: 对应标签
    :param mahalanobis_datas: 待更新类别信息的类
    :return:
    '''
    mahalanobis_datas.add_some_datas(clean_datas.cpu(), clean_labels.cpu())


def get_ood_samples_index(noiy_data, mahalanobis_datas: category_datas, index, threshold):
    ood_index = []
    for i in range(noiy_data.size(0)):
        one_sent = noiy_data[i]
        one_list = []
        for j in range(mahalanobis_datas.label_len):
            d = m_dist(mahalanobis_datas.mean[j], one_sent, mahalanobis_datas.cov_rev[j])
            one_list.append(d.item())
        if min(one_list) > threshold:
            ood_index.append(index[i])
    return torch.LongTensor(ood_index)  # (noisy_size,class_num)


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
    max_len = 128
    data_dir=cfg.train_data
    test_data_dir=cfg.test_data
    #data_dir = r'C:\project\zyz\co_lnl_ood\dataset\20newsgroup\train_noisy_asym0.6.csv'
    #test_data_dir = r'C:\project\zyz\co_lnl_ood\dataset\20newsgroup\test.csv'
    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16
#what are you doing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    '''
    net1 = TransformerModel(cfg.dict_len,cfg.d_model,cfg.trans_n_head,cfg.trans_n_layer)
    net2 = TransformerModel(cfg.dict_len,cfg.d_model,cfg.trans_n_head,cfg.trans_n_layer)
    '''
    net1 = bert_model.bone_BERT(n_classes, pretrained_model_name=cfg.pretrainedmodel)

    optimizer1 = build_sgd_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay, net=net1)

    # optimizer1 = build_adam_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay)
    # optimizer2 = build_adam_optimizer(net2.parameters(), cfg.lr, cfg.weight_decay)
    net1 = net1.to(device)


    net1, optimizer1 = amp.initialize(net1, optimizer1, opt_level=opt_lvl, keep_batchnorm_fp32=None,
                                      loss_scale=None, verbosity=0)

    '''
    net1 = ResNet(arch=cfg.net1, num_classes=n_classes, pretrained=True)
    optimizer1 = build_sgd_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay)
    net1, optimizer1 = amp.initialize(net1.to(device), optimizer1, opt_level=opt_lvl, keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)
    net2 = ResNet(arch=cfg.net2, num_classes=n_classes, pretrained=True)
    optimizer2 = build_sgd_optimizer(net2.parameters(), cfg.lr, cfg.weight_decay)

    net1 = transformer_model.TransformerModel(21128, 768, 4, 256, 4, maxlen=max_len)
    optimizer1 = build_sgd_optimizer(net1.parameters(), cfg.lr, cfg.weight_decay)
    net1, optimizer1 = amp.initialize(net1.to(device), optimizer1, opt_level=opt_lvl, keep_batchnorm_fp32=None,
                                      loss_scale=None, verbosity=0)
    net2 = transformer_model.TransformerModel(21128, 768, 4, 256, 4, maxlen=max_len)
    optimizer2 = build_sgd_optimizer(net2.parameters(), cfg.lr, cfg.weight_decay)
    # 混合精度计算加速
    net2, optimizer2 = amp.initialize(net2.to(device), optimizer2, opt_level=opt_lvl, keep_batchnorm_fp32=None,
                                      loss_scale=None, verbosity=0)
    '''
    # 学习率变化
    lr_plan = make_lr_plan(0.1, cfg.stage1, cfg.epochs)

    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net1.__repr__())
        f.write('\n\n---------------------------\n\n')


    # drop rate scheduler ----------------------------------------------------------------------------------------------------------------------------
    T_k = cfg.stage1
    final_drop_rate = 0.25
    final_ldl_rate = cfg.ldl_rate
    drop_rate_scheduler = np.ones(cfg.epochs) * final_drop_rate
    drop_rate_scheduler[:T_k] = np.linspace(0, final_drop_rate, T_k)
    drop_rate_scheduler[T_k:cfg.epochs] = np.linspace(final_drop_rate, final_ldl_rate, cfg.epochs - T_k)

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    train_data = load_dataset(data_dir, batch_size=cfg.batch_size, pretrainedmodel=cfg.pretrainedmodel, max_len=max_len,model='train',use_cache=cfg.use_cache)
    test_data = load_dataset(test_data_dir, batch_size=cfg.batch_size,pretrainedmodel=cfg.pretrainedmodel,max_len=max_len,model='test',use_cache=cfg.use_cache)

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss1 = AverageMeter()
    train_f1= get_f1()
    iter_time = AverageMeter()

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    best_f1, best_f2 = 0.0, 0.0
    best_epoch1, best_epoch2 = 0, 0


    flag = [0, 0, 0]
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()
        train_loss1.reset()
        train_f1.reset()


        net1.train()

        # adjust_lr(optimizer1, lr_plan[epoch])
        # adjust_lr(optimizer2, lr_plan[epoch])
        optimizer1.zero_grad()

        print(f'开始 epoch {epoch} : 开始训练  ')
        with tqdm(total=train_data.total_batch) as pbar:
            # train this epoch
            for it, sample in enumerate(train_data.get_batch_data()):
                pbar.update(1)
                s = time.time()

                # optimizer1.zero_grad()
                # optimizer2.zero_grad()


                x1 = sample['x'].to(device)
                y0 = torch.tensor(sample['y']).to(device)
                y = get_smoothed_label_distribution(y0, nc=n_classes, epsilon=cfg.epsilon)
                padding_mask = torch.tensor(sample['mask'])
                padding_mask = padding_mask.to(0)
                if x1.size(0)==0:
                    continue
                output1 = net1(x1, attention_mask=padding_mask)


                prob1 = output1['prob']


                loss1 = cross_entropy(prob1, y)

                p1 = torch.argmax(F.softmax(prob1.detach()), dim=1).tolist()

                train_f1.add_some_result(y0.tolist(), p1)


                train_now_f1 = train_f1.cal_f1_score()

                train_loss1.update(loss1.item(), x1.size(0))


                if cfg.use_fp16:
                    with amp.scale_loss(loss1, optimizer1) as scaled_loss1:
                        scaled_loss1.backward()

                else:
                    loss1.backward()


                optimizer1.step()

                optimizer1.zero_grad()

                '''
                if epoch >= cfg.stage1:

                    y_t1.data.sub_(cfg.lmd * y_t1.grad.data)
                    y_t2.data.sub_(cfg.lmd * y_t2.grad.data)
                    labels2learn1[indices, :] = y_t1.detach().clone().cpu().data
                    labels2learn2[indices, :] = y_t2.detach().clone().cpu().data
                    del y_t1, y_t2'''

                iter_time.update(time.time() - s, 1)

                if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (
                        it + 1 == len(train_data.label_list)):
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                    mem = torch.cuda.memory_reserved() / 2 ** 30
                    console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                      f"Iter:[{it + 1:>4d}/{len(train_data.label_list):>4d}]  " \
                                      f"Train Accuracy 1:[{train_now_f1['avg']:6.2f}]  " \
                                      f"Loss 1:[{train_loss1.avg:4.4f}]  " \
                                      f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                      f"{iter_time.avg:6.2f} sec/iter"
                    logger.debug(console_content)

        print(f' epoch {epoch} : 开始测试  ')

        # evaluate this epoch
        test_f1, sss1 = evaluate(test_data, net1, device, "net1")


        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch1 = epoch + 1
            f = open('./pred_'+data_dir.split("\\")[-1]+'_true.txt', 'a', encoding='utf-8')
            f.write('best f1: ')
            f.write(str(best_f1))
            f.write('\n matrix :')
            f.write(str(sss1['mtx']))
            f.write('\n acc :')
            f.write(str(sss1['acc']))
            f.write('\n recall :')
            f.write(str(sss1['recall']))
            f.write('=====================================================================')
            f.close()
            torch.save(net1.state_dict(), f'{result_dir}/net1_best_epoch.pth')


        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss(1/2): ({train_loss1.avg:>6.4f}) | '
                    f'train accuracy(1/2): ({train_now_f1["avg"]:>6.3f}) | '
                    f'test accuracy(1/2): ({test_f1:>6.3f}) | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy(1/2): ({best_f1:6.3f}) @ epoch: ({best_epoch1:03d})')
        #plot_results_cotraining(result_file=f'{result_dir}/log.txt')


    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    best_accuracy = max(best_f1, best_f2)
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log_prefix', type=str, default='co_lnl_ood')
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=800.00)
    parser.add_argument('--net1', type=str, default='resnet18')
    parser.add_argument('--net2', type=str, default='resnet18')
    parser.add_argument('--train_data', type=str, default='')
    parser.add_argument('--test_data', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--stage1', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lmd', type=float, default=200)
    parser.add_argument('--ldl_rate', type=float, default=0.1)
    parser.add_argument('--phi', type=float, default=0.4)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.1)
    """
    --config config/20newsgroup.cfg --gpu 0 --net1 bert --net2 bert --use_cache False --train_data C:\project\zyz\co_lnl_ood\dataset\20newsgroup\train_noisy_asym0.6.csv --test_data  C:\project\zyz\co_lnl_ood\dataset\20newsgroup\test.csv
    """
    #data_dir = r'C:\project\zyz\co_lnl_ood\dataset\20newsgroup\train_noisy_asym0.6.csv'
    #test_data_dir = r'C:\project\zyz\co_lnl_ood\dataset\20newsgroup\test.csv'
    parser.add_argument('--use_cache', type=bool, default=False)
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

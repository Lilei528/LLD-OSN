import json
import time

import torch
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.utils import get_f1


def accuracy(y_pred, y_actual, topk=(1, ), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res

import torch.nn.functional as F
def evaluate(dataloader, model, dev, netname,topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    print(f'开始评估{netname}')


    model.eval()
    test_f1 =get_f1()

    prb_list=[]
    y_list=[]

    with torch.no_grad():
        with tqdm(total=dataloader.total_batch) as pbar:
            for _, sample in enumerate(tqdm(dataloader.get_batch_data(), ncols=100, ascii=' >')):
                pbar.update(1)
                x = sample['x'].to(dev)
                y = sample['y'].to(dev)
                output = model(x,attention_mask=sample['mask'].to(0))
                logits = output['prob']
                p1= torch.argmax(F.softmax(logits),dim=1).tolist()

                test_f1.add_some_result(y.tolist(), p1)

                prb_list.extend(F.softmax(logits).tolist())
                y_list.extend(y.tolist())

            test_f2 = test_f1.cal_f1_score()
            a=torch.tensor(prb_list)
            b=torch.tensor(y_list)
            max_probs, _ = torch.max(a, dim=1)

            # 筛选出预测正确的样本
            correct_samples = (b == torch.argmax(a, dim=1)).float()

            # 计算预测正确的样本的平均置信度
            average_confidence_correct = torch.sum(max_probs * correct_samples)/torch.sum(correct_samples)
            incorrect_samples = 1 - correct_samples

            # 计算预测错误的样本的平均置信度
            average_confidence_incorrect = torch.sum(max_probs * incorrect_samples) / torch.sum(incorrect_samples)

            sss={'incof':average_confidence_incorrect.item(),'cof':average_confidence_correct.item()}


    return test_f2['avg'],sss
def evaluate_ood(dataloader, model, dev, netname,topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    print(f'开始计算分布外置信{netname}')


    model.eval()
    aa=[]


    with torch.no_grad():
        with tqdm(total=dataloader.total_batch) as pbar:
            for _, sample in enumerate(tqdm(dataloader.get_batch_data(), ncols=100, ascii=' >')):
                pbar.update(1)
                x = sample['x'].to(dev)
                output = model(x,attention_mask=sample['mask'].to(0))
                logits = output['prob']
                aa.extend(logits.tolist())


        a=F.softmax(torch.tensor(aa))
        max_probs, _ = torch.max(a, dim=1)

        # 计算最大概率的总和
        total_max_prob = torch.sum(max_probs)

        # 计算平均置信度
        num_samples = a.size(0)
        average_confidence = total_max_prob / num_samples

    return average_confidence.item()

def evaluate_text(dataloader,max_len,tokenizer, model, dev, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_accuracy = AverageMeter()
    test_accuracy.reset()
    with torch.no_grad():
        for it, sample in enumerate(dataloader.batch_iter(padding_len=max_len, tokenizer=tokenizer)):

            x = torch.tensor(sample['data'][0]).to(dev)
            y = torch.tensor(sample['label']).to(dev)

            padding_mask = torch.tensor(sample['mask'])
            padding_mask = padding_mask.to(0)
            output = model(x,src_padding_mask=padding_mask)
            logits = output['logits']
            acc = accuracy(logits, y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return test_accuracy.avg


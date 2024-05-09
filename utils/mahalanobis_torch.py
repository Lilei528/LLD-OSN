#判断是否是正定矩阵
import numpy as np
import torch



def m_dist(data_mean,x,cov_rev):
    '''
    :param data_mean:  一个样本集的均值
    :param x: 待检测的潜在离群点
    :param x: 一个样本集的协方差逆矩阵
    :return:
    '''

    diff = x-data_mean#  求差
    diff = torch.unsqueeze(diff, dim=0)
    res = torch.mm(torch.mm(diff,cov_rev) , diff.T )   #  最后根据公式组合在一起
    return torch.sqrt(res)


def get_mean(datas):
    return  torch.mean(datas, dim=0)

def get_cov_rev(datas):
    '''
    :param datas: batch*feat
    :return:
    '''
    if datas.size(0)<datas.size(1):
        return torch.eye(datas.size(1))
    else:
        cov = torch.cov(datas.t())
        try:
            cov_rev = torch.inverse(cov)
        except:
            return torch.eye(datas.size(1))
    return cov_rev
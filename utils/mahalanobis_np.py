import numpy as np
import math
'''
numpy实现版本
'''
def m_dist(data_mean,x,cov_rev):
    '''
    :param data_mean:  一个样本集的均值
    :param x: 待检测的潜在离群点
    :param x: 一个样本集的协方差逆矩阵
    :return:
    '''

    diff = x-data_mean#  求差
    res = math.sqrt(np.dot(diff,cov_rev).dot(diff.T))#  最后根据公式组合在一起
    return res

def get_mean(datas):
    return np.array(datas.mean(axis=0))

def get_cov_rev(datas):
    '''
    :param datas:  sample_size*feat_len
    :return: cov_rev feat_len*feat_len
    '''
    if datas.shape[0]<datas.shape[1]:
        print('类别样本少于特征！这次设协方差为单位矩阵')
        return np.eye(datas.shape[1])
    else:
        cov=np.cov(datas.T)
        return  np.linalg.inv(cov)  #注意参数data.T因为函数输入为（特征*batch）




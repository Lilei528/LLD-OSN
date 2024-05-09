import numpy
import numpy as np
import torch

from mahalanobis_np import *
class category_datas():
    def __init__(self,n_class,category_keep_num):
        '''
        :param n_class:   样本批注标签类别总数
        :param category_keep_num:   计算马氏距离时使用样本集的每个类别的样本最大数量
        '''
        self.datas=[NumpyQueue(category_keep_num) for i in range(n_class)]
        self.cov_rev=[None]*n_class
        self.mean=[None]*n_class

    def add_some_datas(self,new_datas,data_lables,update_cm=True):
        '''
        :param new_datas: tensor  size*feat
        :param data_lables: tensor    size
        :return:
        '''
        alb=data_lables.tolist()
        for index,i in enumerate(alb):
            self.datas[i].add_one_elements(new_datas[index])
        changed_label=set(alb)
        if update_cm:
            self.update_cov_mean(update_label=changed_label)
    def update_cov_mean(self,update_label=None):
        if update_label==None:
            for i in range(len(self.datas)):
                s=self.datas[i].get_all_elements()
                self.cov_rev[i]=get_cov_rev(s)
                self.mean[i] = get_mean(s)
        else:
            for i in update_label:
                s=self.datas[i].get_all_elements()
                self.cov_rev[i]=get_cov_rev(s)
                self.mean[i] = get_mean(s)
    def update_mean(self,update_label=None):
        if update_label==None:
            for i in range(len(self.datas)):
                s=self.datas[i].get_all_elements()
                self.mean[i] = get_mean(s)
        else:
            for i in update_label:
                s=self.datas[i].get_all_elements()
                self.mean[i] = get_mean(s)

    def update_cov(self,update_label=None):
        if update_label == None:
            for i in range(len(self.datas)):
                s = self.datas[i].get_all_elements()
                self.cov_rev[i] = get_cov_rev(s)
        else:
            for i in update_label:
                s = self.datas[i].get_all_elements()
                self.cov_rev[i] = get_cov_rev(s)



class NumpyQueue:
    def __init__(self, max_length):
        """
        初始化一个有最大长度的队列。
        :param max_length: 队列的最大长度
        """
        self.max_length = max_length
        self.queue = [] # queue中存储的tensor
    def add_one_elements(self, element):
        """
        一次性添加多个元素到队列中。如果队列长度超过最大长度，将移除最早加入的元素。
        :param elements: 要添加的元素
        """

        if len(self.queue) >= self.max_length:
            self.queue.pop(0)  # 移除最早加入的元素
        self.queue.append(element)

    def add_elements(self, *elements):
        """
        一次性添加多个元素到队列中。如果队列长度超过最大长度，将移除最早加入的元素。
        :param elements: 要添加的元素（numpy 数组）
        """
        for element in elements:
            if len(self.queue) >= self.max_length:
                self.queue.pop(0)  # 移除最早加入的元素
            self.queue.append(element)

    def get_all_elements(self):
        """
        获取队列中的全部元素。
        :return: 队列中的所有元素（tensor）
        """
        return np.array(self.queue)

    def get_element_count(self):
        """
        获取队列中的元素数量。
        :return: 队列中的元素数量（整数）
        """
        return len(self.queue)

# test

class_a=numpy.array(torch.Tensor( [

    [0.004,-0.126,0.21],
    [0.212,0.451,-0.1522],
    [0.214,0.459,0.451]
]  ))
class_b=numpy.array(torch.Tensor( [
    [-0.451,0.5124,0.2651],
    [0.451,0.124,-0.1],
    [0.451,-0.454,0.45121],
    [0.452123,0.45122,0.45121]
]  ))
x=torch.Tensor(
    [0.211,0.312,-0.214]
)
mc=category_datas(2,3)
mc.add_some_datas(class_a,torch.tensor([0,0,0]))
mc.add_some_datas(class_b,torch.tensor([1,1,1]))

d=m_dist(mc.mean[1],numpy.array(x),mc.cov_rev[1])
print(d)


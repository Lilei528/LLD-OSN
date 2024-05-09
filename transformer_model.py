import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.utils import init_weights
import math


# unsqueeze(index) 在index增加一个大小为1的维度
# squeeze(index)删除index或者所有的维度为1的维度
# reshape()改成想要的维度，-1表示其他剩余的组成
# t()转置
# permute().contiguous() 交换多个维度,并让新的矩阵在内存中连续
# transpose().contiguous()  交换两个维度,并让新的矩阵在内存中连续

class PositionalEncoding(nn.Module):
    "实现PE功能"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数列
        pe = pe.unsqueeze(0).transpose(0,1)       # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  输入x的维度：[max_len,batch_size,d_model ]
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)   # 这里是Embedding与Positional Encoding相结合
        return self.dropout(x)

'''
class PositionalEncoding(nn.Module):
    # d_model词向量的长度，max_len句子长度
    def __init__(self, d_model,max_len, dropout=0.1, ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
'''

from torch.nn import TransformerEncoderLayer,TransformerEncoder
# ntoken:字典大小   ninp:每个词的维度   nhead：多少个头   nhid：encoder输出的维度  nlayers：encoder层数  maxlen，class——len
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nlayers,
                 dropout=0.5,maxlen=256,class_len=6,dim_feedforward=2048,classifier='mlp'):
        # ntokens 词典大小,ninp = emsize 词嵌入维度 200, nhead 模型head数量 2,  nlayers Encoder的层数 2, dropout=0.2
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp,dropout)

        encoder_layers = TransformerEncoderLayer(ninp, nhead,dim_feedforward=dim_feedforward, dropout= dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)

        self.ntoken=ntoken
        self.ninp=ninp

        self.classfier_head = nn.Linear(in_features=ninp * maxlen, out_features=class_len)

        self.ood_head= nn.Linear(in_features=ninp * maxlen, out_features=2)
        """
        self.classfier_head = nn.Linear(in_features=ninp, out_features=class_len)
        self.ood_head= nn.Linear(in_features=ninp , out_features=2)

        """

        self.init_weights()
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # 线性层权重初始化
        self.classfier_head.bias.data.zero_()
        self.classfier_head.weight.data.uniform_(-initrange, initrange)

        self.ood_head.bias.data.zero_()
        self.ood_head.weight.data.uniform_(-initrange, initrange)

    #mask大小是(batch_size,max_len,max_len)在多头注意力中用到的,-1或-inf
    def _generate_square_subsequent_mask(self, src, lenths):
        '''
        padding_mask
        src:batch_size,max_len
        lenths:[lenth1,lenth2...]
        '''

        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(0), src.size(1)) == 1
        for i in range(len(lenths)):
            lenth = lenths[i]
            for j in range(lenth):
                mask[i][j] = False

        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    # input shape: batch,max_len,vocab_len,过程中转换input的矩阵为max_len,batch,vocab_len
    # length:每个句子的实际长度（因为padding），为 list：  [ 218,122,...  ] (batch_size*1)
    def forward(self, src, length=[],src_padding_mask=None):
        """
        src: batch_size*max_len

        """
        if len(length)==0 and src_padding_mask==None:
            warnings.WarningMessage("没有padding数据，无法遮盖padding")
            exit(-1)


        # torch.nn.TransformerEncoder的forward时的src维度默认为（maxlen，batchsize，ninp），src_key_padding_mask为（batchsize，maxlen）
        if src_padding_mask==None:
            # padding_mask  =batchsize，maxlen
            padding_mask=self._generate_square_subsequent_mask(src,length)

        src = self.embedding(src) * math.sqrt(self.ninp)
        #src = src.transpose(0, 1).contiguous()
        #src (maxlen*batchsize*ninp)
        src = self.pos_encoder(src.transpose(0, 1))

        output = self.transformer_encoder(src,src_key_padding_mask=padding_mask)
        #output (maxlen*batchsize*ninp)
        output=output.transpose(0, 1).contiguous()
        '''
        output = output[:,0,:]
        output = output.view(output.shape[0], -1)

        logic = self.classfier_head(output)

        ood_logic=self.ood_head(output)
        return {'logic': logic, 'ood_logic': ood_logic , 'feat': output }
        '''

        #output (batchsize,maxlen*ninp)
        output = output.view(output.shape[0], -1)

        #output (batchsize,label_len)
        logic = self.classfier_head(output)

        ood_logic=self.ood_head(output)
        return {'logic': logic, 'ood_logic': ood_logic , 'feat': output }





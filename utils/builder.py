
import torch.optim as optim


def build_sgd_optimizer(params, lr, weight_decay, net,nesterov=True):
    params = list(net.named_parameters())
    no_decay = ['bias','LayerNorm']
    other = ['id_fc','ood_fc']

    no_main = no_decay + other

    param_group = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-7,'lr':1e-5,'mname':'bert'},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':1e-5,'mname':'bert_nodecay'},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':0.01,'mname':'fc_nodecay'},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':1e-7,'lr':0.01,'mname':'fc_decay'},

    ]

    return optim.SGD(param_group, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)
    #optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)
def build_adam_optimizer(params, lr, weight_decay):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay)

from torch import nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.sparse


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))    #初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G.to_sparse(), x)
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, momentum=0.1):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.batch_normalzation2 = nn.BatchNorm1d(n_hid, momentum=momentum)

    def forward(self, x, G):
        x = self.batch_normalzation1(x)
        x = self.hgc1(x, G)
        x = self.batch_normalzation2(x)
        x = F.relu(x)
        x = self.batch_normalzation2(x)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

class compute_G(nn.Module):
    def __init__(self, W):
        super(compute_G, self).__init__()
        self.W = Parameter(W)
    def forward(self, DV2_H, invDE_HT_DV2):
        w = torch.diag(self.W)
        G = torch.sparse.mm(w.to_sparse(), invDE_HT_DV2)
        #G = DV2_H.matmul(G)
        G = torch.sparse.mm(DV2_H.to_sparse(), G)
        return G

class compute_G_2(nn.Module):
    def __init__(self, W):
        super(compute_G_2, self).__init__()
        self.W = Parameter(W)
    def forward(self, H, invDE_HT):
        w = torch.diag(self.W)
        DV = torch.sparse.mm(H.to_sparse(),w)
        DV = torch.sum(DV, dim=1)
        DV2 = torch.diag(torch.pow(DV, -0.5))
        #DV2_H = DV2 @ H @ w @ invDE_HT @ DV2
        DV2_H = torch.sparse.mm(DV2.to_sparse(), H)
        invDE_HT_DV2 = torch.sparse.mm(invDE_HT.to_sparse(), DV2)
        G = torch.sparse.mm(w.to_sparse(), invDE_HT_DV2)
        G = torch.sparse.mm(DV2_H.to_sparse(), G)

        return G


class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, W, dropout=0.5, momentum=0.1):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.computeG = compute_G(W)
        self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.batch_normalzation2 = nn.BatchNorm1d(n_hid, momentum=momentum)

    def forward(self, x, DV2_H, invDE_HT_DV2):
        x = self.batch_normalzation1(x)
        G = self.computeG(DV2_H, invDE_HT_DV2)
        x = self.hgc1(x, G)
        x = self.batch_normalzation2(x)
        x = F.relu(x)
        x = self.batch_normalzation2(x)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

if __name__ == '__main__':
    model = torch.load('./model_dict/parameter.pkl')
    A = model['computeG.W']
    print(A)
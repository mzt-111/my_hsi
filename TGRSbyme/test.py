import numpy as np
import scipy.sparse as sp
import torch
import torch.sparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device = torch.device('cuda:0')
a = torch.eye(4000)
a.to(device)
b = torch.mm(a, a)

ad
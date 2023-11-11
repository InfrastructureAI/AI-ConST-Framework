import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.cluster
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def cluster(data, k, temp, num_iter, init, cluster_temp):
    cuda0 = torch.cuda.is_available()

    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init

    data = data / (data.norm(dim=1)[:, None] +
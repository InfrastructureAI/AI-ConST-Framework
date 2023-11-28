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

    data = data / (data.norm(dim=1)[:, None] + 1e-6)  # prevent zero-division loss with 1e-6
    for t in range(num_iter):

        mu = mu / (mu.norm(dim=1)[:, None] + 1e-6) #prevent zero-division with 1e-6

        dist = torch.mm(data, mu.transpose(0,1))

        # cluster responsibilities via softmax
        r = F.softmax(cluster_temp*dist, dim=1)
        #
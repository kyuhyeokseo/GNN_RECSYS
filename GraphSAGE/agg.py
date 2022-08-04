import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    def __init__(self, features, feat_dim, num_sample, cuda):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.num_sample = num_sample
        self.feat_dim = feat_dim

    def forward(self, nodes, to_neighs):
        _set = set
        _sample = random.sample
        samp_neighs = [ _set( _sample(to_neigh, self.num_sample,) ) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs ]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        if self.cuda: embed_matrix = embed_matrix.cuda()

        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.cuda: mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        to_feats = mask.mm(embed_matrix)

        return to_feats



class meanPoolAggregator(nn.Module):
    def __init__(self, features, feat_dim, num_sample, cuda):
        super(meanPoolAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.num_sample = num_sample
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes, to_neighs):
        _set = set
        _sample = random.sample
        samp_neighs = [ _set( _sample(to_neigh, self.num_sample,) ) if len(to_neigh) >= self.num_sample
                        else to_neigh for to_neigh in to_neighs ]

        n = len(nodes)

        out = torch.zeros(n, self.feat_dim)

        for i in range(n):
            embed_matrix = self.features(torch.LongTensor(list(samp_neighs[i])))
            if self.cuda:
                embed_matrix = embed_matrix.cuda()
            temp = self.relu(self.fc1(embed_matrix))
            out[i, :] = torch.mean(temp, dim=0)

        return out



class GCNAggregator(nn.Module):
    def __init__(self, features, feat_dim, num_sample, cuda):
        super(GCNAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.num_sample = num_sample
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes, to_neighs):
        _set = set
        _sample = random.sample
        samp_neighs = [ _set( _sample(to_neigh, self.num_sample,) ) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs ]

        samp_neighs = [ samp_neigh.union(_set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        if self.cuda:
            embed_matrix = embed_matrix.cuda()

        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)

        mask = mask.div(num_neigh)

        to_feats = mask.mm(embed_matrix)

        return to_feats


class LSTMAggregator(nn.Module):
    def __init__(self, features, feat_dim, num_sample, cuda):
        super(LSTMAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.num_sample = num_sample
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(feat_dim, feat_dim, bidirectional=True, batch_first=True)


    def forward(self, nodes, to_neighs):
        _set = set
        _sample = random.sample
        samp_neighs = [ _set( _sample(to_neigh, self.num_sample,) ) if len(to_neigh) >= self.num_sample
                        else to_neigh for to_neigh in to_neighs ]

        out = torch.zeros(len(nodes), 2 * self.feat_dim)
        for i in range(len(nodes)):
            embed_matrix = self.features(torch.LongTensor(list(samp_neighs[i])))
            if self.cuda:
                embed_matrix = embed_matrix.cuda()

            perm = np.random.permutation(np.arange(embed_matrix.shape[0]))
            embed_matrix = embed_matrix[perm, :]
            embed_matrix = embed_matrix.unsqueeze(0)

            temp, _ = self.lstm(embed_matrix)
            temp = temp.squeeze(0)
            temp = torch.sum(temp, dim=0)

            out[i, :] = torch.mean(temp, dim=0)

        return out

class torch_LSTMAggregator(nn.Module):
    def __init__(self, features, feat_dim, num_sample, cuda):
        super(torch_LSTMAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.num_sample = num_sample
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(feat_dim, feat_dim, bidirectional=True, batch_first=True)


    def forward(self, nodes, to_neighs):
        _set = set
        _sample = random.sample
        samp_neighs = [ _set( _sample(to_neigh, self.num_sample,) ) if len(to_neigh) >= self.num_sample
                        else to_neigh for to_neigh in to_neighs ]

        out = torch.zeros(len(nodes), 2 * self.feat_dim)
        for i in range(len(nodes)):
            embed_matrix = self.features(torch.LongTensor(list(samp_neighs[i])))
            if self.cuda:
                embed_matrix = embed_matrix.cuda()

            perm = np.random.permutation(np.arange(embed_matrix.shape[0]))
            embed_matrix = embed_matrix[perm, :]
            embed_matrix = embed_matrix.unsqueeze(0)

            temp, _ = self.lstm(embed_matrix)
            temp = temp.squeeze(0)
            temp = torch.sum(temp, dim=0)

            out[i, :] = torch.mean(temp, dim=0)

        return out
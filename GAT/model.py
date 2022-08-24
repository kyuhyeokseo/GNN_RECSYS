import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter( torch.empty(in_features, out_features) )
        nn.init.xavier_uniform_( self.W.data, gain = np.sqrt(2) )
        self.a = nn.Parameter( torch.empty(2*out_features , 1) )
        nn.init.xavier_uniform_( self.a.data, gain = np.sqrt(2) )

        self.LeakyReLU = nn.LeakyReLU(self.alpha)


    def forward(self, h, adj):

        # (N, in_features) -> (N, out_features)
        Wh= torch.mm(h, self.W)

        attention = self._prepare_attentional_mechanism_input(Wh, adj)

        # To make h_prime vector
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh, adj):

        Wh1 = torch.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = torch.matmul(Wh, self.a[:self.out_features, :])

        e = torch.zeros(Wh.shape[0], Wh.shape[0]) # Initializing

        # sparse로 바꿔서 adj 해당할때만 e에 기록하게끔 수정 -> 시간문제
        cx = coo_matrix(adj)
        for u, i, v in zip(cx.row, cx.col, cx.data):
            e[u,i] = Wh1[u,0] + Wh2[i,0]

        return self.LeakyReLU(e)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [ GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads) ]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):

        x = F.dropout(x, self.dropout)
        print("line75")
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print("line77")
        x = F.dropout(x, self.dropout)
        print("line79")
        x = F.elu(self.out_att(x, adj))
        print("line81")

        return F.log_softmax(x, dim=1)
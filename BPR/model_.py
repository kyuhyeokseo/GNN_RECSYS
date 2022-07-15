import torch
import torch.nn as nn
from torch import relu, FloatTensor

class BPR_MF(nn.Module):
    def __init__(self, neg_item, config):
        super(BPR_MF, self).__init__()

        self.n_user = config.n_user
        self.n_item = config.n_item
        self.embed_dim = config.embed_dim

        self.embed_u = nn.utils.weight_norm(nn.Embedding(self.n_user, self.embed_dim))
        self.embed_v = nn.utils.weight_norm(nn.Embedding(self.n_item, self.embed_dim))
        self.embed_u.weight.data.normal_(mean=0, std=0.01)
        self.embed_v.weight.data.normal_(mean=0, std=0.01)
        self.config = config
        self.sig = nn.Sigmoid()

    def forward(self, u, i, j):

        # Compute
        W_u = self.embed_u(u)
        H_i = self.embed_v(i)
        H_j = self.embed_v(j)

        prediction_i = torch.sum(W_u * H_i)
        prediction_j = torch.sum(W_u * H_j)

        return prediction_i, prediction_j


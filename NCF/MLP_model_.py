import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLP(nn.Module):
    def __init__(self, n_user, n_item, config):
        super(MLP, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        self.config = config

        self.embed_u = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_i = nn.Embedding(self.n_item, self.embed_dim)

        self.MLP1 = nn.Linear(self.embed_dim * 2, self.hidden_dim[0])
        self.MLP2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.MLP3 = nn.Linear(self.hidden_dim[1], self.hidden_dim[2])

        self.H = nn.Linear(self.hidden_dim[2], 1)
        self.sigmoid = nn.Sigmoid()

        # 각 embedding, linear가 가우시안 분포를 다르도록 initializing
        self._init_weight_()


    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, 0.01)



    def forward(self, user, item):

        P_u = self.embed_u(user)
        Q_i = self.embed_i(item)

        Z = torch.cat( (P_u, Q_i), dim=1)

        Layer1 = relu(self.MLP1(Z))
        Layer2 = relu(self.MLP2(Layer1))
        Layer3 = relu(self.MLP3(Layer2))

        Calculate = self.H(Layer3)
        Calculate = self.sigmoid(Calculate)

        return Calculate

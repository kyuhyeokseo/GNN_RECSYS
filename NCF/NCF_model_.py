import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

class NCF(nn.Module):
    def __init__(self, n_user, n_item, config):
        super(NCF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.config = config

        self.embed_u_mlp = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_i_mlp = nn.Embedding(self.n_item, self.embed_dim)

        self.MLP1 = nn.Linear(self.embed_dim * 2, self.hidden_dim[0])
        self.MLP2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.MLP3 = nn.Linear(self.hidden_dim[1], self.hidden_dim[2])

        self.embed_u_gmf = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_i_gmf = nn.Embedding(self.n_item, self.embed_dim)

        self.H = nn.Linear(self.embed_dim + self.hidden_dim[2], 1)
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


    def from_pretrained(self, path, model_):
        if self.config.pretrained:
            gmf_pretrained = torch.load(f'{path}/pretrained_GMF.pth')
            mlp_pretrained = torch.load(f'{path}/pretrained_MLP.pth')

            gmf_H = nn.Linear(self.embed_dim, 1)
            mlp_H = nn.Linear(self.hidden_dim[2], 1)

            for name, weight in gmf_pretrained.items():
                if name == 'embed_u.weight':
                    model_.embed_u_mlp.weight.data = weight
                elif name == 'embed_i.weight':
                    model_.embed_i_mlp.weight.data = weight
                elif name == 'H.weight':
                    gmf_H = weight

            for name, weight in mlp_pretrained.items():
                if name == 'embed_u.weight':
                    model_.embed_u_mlp.weight.data = weight
                elif name == 'embed_i.weight':
                    model_.embed_i_mlp.weight.data = weight
                elif name == 'MLP1.weight':
                    model_.MLP1.weight.data = weight
                elif name == 'MLP2.weight':
                    model_.MLP2.weight.data = weight
                elif name == 'MLP3.weight':
                    model_.MLP3.weight.data = weight
                elif name == 'H.weight':
                    mlp_H = weight

            new_H = torch.cat( ( gmf_H * 0.5, mlp_H * 0.5), dim=1 )
            model_.H.weight.data= new_H

            return model_

        else:
            print('Use no pretrained model')
            return model_


    def forward(self, user, item):

        P_u_gmf = self.embed_u_gmf(user)
        Q_i_gmf = self.embed_i_gmf(item)

        Calculate_gmf = P_u_gmf * Q_i_gmf

        P_u_mlp = self.embed_u_mlp(user)
        Q_i_mlp = self.embed_i_mlp(item)

        Z = torch.cat( (P_u_mlp, Q_i_mlp), dim=1)

        Layer1 = relu(self.MLP1(Z))
        Layer2 = relu(self.MLP2(Layer1))
        Layer3 = relu(self.MLP3(Layer2))

        Calculate_mlp = Layer3

        Calculate = torch.cat( (Calculate_gmf, Calculate_mlp), dim=1 )
        Calculate = self.H(Calculate)
        Calculate = self.sigmoid(Calculate)

        return Calculate

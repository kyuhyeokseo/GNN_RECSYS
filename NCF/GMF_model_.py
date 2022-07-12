
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_user, n_item, config):
        super(GMF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.embed_dim = config.embed_dim
        self.config = config

        self.embed_u = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_i = nn.Embedding(self.n_item, self.embed_dim)

        self.H = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, user, item, binary):

        P_u = self.embed_u(user)
        Q_i = self.embed_i(item)

        Calculate = P_u * Q_i
        Calculate = self.H(Calculate)
        Calculate = self.sigmoid(Calculate)

        return Calculate

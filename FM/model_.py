import torch
import torch.nn as nn
from torch import relu, FloatTensor

class FM(nn.Module):
    def __init__(self, data, target, config):
        super(FM, self).__init__()

        self.config = config
        self.n_data = data.shape[0]
        self.n_feature = data.shape[1]

        self.k = config.size

        self.bias = nn.Parameter(torch.tensor([1], dtype=torch.float))
        self.w = nn.Parameter(torch.randn( self.n_feature, dtype=torch.float ))
        self.V = nn.Parameter(torch.randn( ( self.n_feature, self.k ), dtype=torch.float ))

    def forward(self, batch):
        data, target = batch['feature'], batch['value']
        batch_size = self.config.batch_size

        # second Summation
        second = torch.matmul(data, self.w)

        # third Summation
        third = torch.sum((torch.matmul(data, self.V)**2 - torch.matmul(data**2, self.V**2)),dim=1)/2

        return self.bias + second + third


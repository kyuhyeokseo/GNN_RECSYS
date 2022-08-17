import torch
import torch.nn as nn

class PMF(nn.Module):
    def __init__(self, data, user_num, item_num, max_rate, config):
        super(PMF, self).__init__()

        self.data = data
        self.n_user = user_num
        self.n_item = item_num
        self.K = max_rate
        self.config = config

        self.user_embed = nn.Parameter(torch.randn(self.n_user, config.embed_dim, requires_grad=True))

        self.item_embed = nn.Parameter(torch.randn(self.n_item, config.embed_dim, requires_grad=True))


    def forward(self, users_index, items_index, value):

        user_h1 = self.user_embed[users_index , : ]
        item_h1 = self.item_embed[items_index , : ]

        pred = 1/(1 + torch.exp(-((user_h1 * item_h1).sum(0))))

        value = (value - 1) / ( self.K -  1)

        result = pred - value

        return result ** 2
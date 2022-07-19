import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np

class Wide_Deep(nn.Module):
    def __init__(self, cat_cols, new_cols, cont_cols, cat_encoders ,config, wide_cols):
        super(Wide_Deep, self).__init__()

        self.cont_cols = cont_cols
        self.cont_dim = len(self.cont_cols)
        self.new_cols = new_cols
        self.cat_cols = cat_cols
        self.wide_cols = wide_cols

        # For WIDE Part
        self.wide_input_dim = len(self.wide_cols)
        self.wide_W = nn.Linear(self.wide_input_dim, 1)

        # For DEEP Part - 1. category Embedding
        self.cat_dim_list = [len(v.classes_) for k, v in cat_encoders.items()]
        self.embeddings_list = nn.ModuleList([nn.Embedding(v, config.embed_dim) for v in self.cat_dim_list])
        self.embed_dim = len(self.cat_dim_list * config.embed_dim) + self.cont_dim

        # For DEEP Part - 2
        self.deep_Linear1 = nn.Linear(self.embed_dim, 256)
        self.deep_Linear2 = nn.Linear(256, 128)
        self.deep_Linear3 = nn.Linear(128, 64)
        self.deep_Linear4 = nn.Linear(64, 1)

        # Joint
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.FloatTensor([0.5]))


    def forward(self, wide_x, deep_x):

        # WIDE
        wide_y = self.wide_W(wide_x)

        # DEEP
        cont_tensor = deep_x[:, len(self.embeddings_list):]
        cat_tensor = deep_x[:, :len(self.embeddings_list)].long()

        pre_embed = [e(cat_tensor[:, i]) for i, e in enumerate(self.embeddings_list)]
        pre_embed = torch.cat(pre_embed, dim=1)

        deep_y = torch.cat([cont_tensor, pre_embed], dim=1)
        deep_y = relu(self.deep_Linear1(deep_y))
        deep_y = relu(self.deep_Linear2(deep_y))
        deep_y = relu(self.deep_Linear3(deep_y))
        deep_y = relu(self.deep_Linear4(deep_y))

        Calculate = self.sigmoid(self.weight * wide_y + (1-self.weight) * deep_y)

        return Calculate

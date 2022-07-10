import torch
import torch.nn as nn
from torch import relu, FloatTensor

class CML(nn.Module):
    def __init__(self, input_dim, neg_item, config):
        super(CML, self).__init__()
        self.n_user = config.n_user
        self.n_item = config.n_item
        self.input_dim = input_dim
        self.embed_dim = config.embed_dim
        self.neg_sample_size = config.neg_sample_size
        self.margin = config.margin
        self.neg_item = neg_item

        self.W_ij = torch.rand(self.n_user, self.n_item).to(config.device) * 10
        self.embed_u = nn.utils.weight_norm(nn.Embedding(self.n_user, self.embed_dim))
        self.embed_v = nn.utils.weight_norm(nn.Embedding(self.n_item, self.embed_dim))
        self.embed_u.weight.data.normal_(mean=0, std=1/self.embed_dim**0.5)
        self.embed_v.weight.data.normal_(mean=0, std=1/self.embed_dim**0.5)

        self.MLP1 = nn.Linear(self.input_dim, self.input_dim)
        self.MLP2 = nn.Linear(self.input_dim, self.embed_dim)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)

        self.lambda_f = torch.FloatTensor([1]).to(config.device)
        self.lambda_c = torch.FloatTensor([10]).to(config.device)


    def forward(self, batch):
        user, item, neg_item, item_detail = batch['user'], batch['item'], batch['neg_item'], batch['item_detail'].float()
        batch_size = user.size(0)

        # Compute Loss_m
        W_ij = self.W_ij[user, item].unsqueeze(-1)
        U_i = self.embed_u(user)
        V_j = self.embed_v(item)
        V_k = self.embed_v(neg_item)

        D_ij = self.distance_ij(U_i, V_j)
        D_ik = self.distance_ik(U_i, V_k)

        L_m = torch.sum(torch.sum( W_ij * ( self.margin + D_ij - D_ik ), axis = 1 ))

        # Compute Loss_f
        item_detail = self.dropout1(item_detail)
        item_detail = torch.relu(self.MLP1(item_detail))
        item_detail = self.dropout2(item_detail)
        item_detail = torch.relu(self.MLP2(item_detail))
        L_f = torch.sum(torch.sum( (item_detail - V_j)**2, axis=1 ))

        # Compute Loss_c
        C_ij = self.get_mat_C(U_i, V_j, batch_size)
        L_c = ( torch.norm(C_ij, p='fro') - torch.norm(torch.diagonal( C_ij , 0), 2))/batch_size

        impost = torch.sum( ( self.margin + D_ij  - D_ik ) > 0, axis=1)
        self.W_ij[user, item] = torch.log( impost * self.n_item / self.neg_sample_size + 1)

        return L_m + self.lambda_f * L_f + self.lambda_c * L_c


    def distance_ij(self, U_i, V_j):
        return torch.sum((U_i - V_j)**2, axis=1).unsqueeze(-1)

    def distance_ik(self, U_i, V_k):
        return torch.sum( (U_i.unsqueeze(axis=1) - V_k)**2, axis=2)

    def get_mat_C(self, U_i, V_j, batch_size):
        cat_emb = torch.cat((U_i, V_j), axis=0)
        mu = torch.mean(cat_emb, axis=0)
        cat_emb = cat_emb - mu
        C = torch.matmul(cat_emb.T, cat_emb) / batch_size
        return C
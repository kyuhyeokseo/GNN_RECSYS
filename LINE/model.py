import torch
from torch import nn


class LINE(nn.Module):
    def __init__(self, n_node, embed_dim):
        super(LINE, self).__init__()

        self.embed_1 = nn.Embedding(n_node, embed_dim)
        self.embed_2 = nn.Embedding(n_node, embed_dim)
        self.embed_2_context = nn.Embedding(n_node, embed_dim)


    def forward(self, batch_data):

        pos_edge = batch_data['pos']
        pos_edge_w = batch_data['pos_w']
        neg_edge = batch_data['neg']

        pos_from_, pos_to_ = pos_edge[:, 0], pos_edge[:, 1]
        neg_from_, neg_to_ = neg_edge[:, 0], neg_edge[:, 1]

        return {
            'first': self.first_loss(pos_from_, pos_to_, pos_edge_w),
            'second': self.second_loss(pos_from_, pos_to_, neg_from_, neg_to_)
        }


    def first_loss(self, pos_from_, pos_to_, pos_edge_w):

        u_i = self.embed_1(pos_from_)
        u_j = self.embed_1(pos_to_)

        p_vivj = torch.sigmoid(torch.sum(u_i * u_j, dim=1))
        loss = -torch.sum(pos_edge_w * torch.log(p_vivj))

        return loss

    def second_loss(self, pos_from_, pos_to_, neg_from_, neg_to_):

        u_i = self.embed_2(pos_from_)
        u_jp = self.embed_2_context(pos_to_)
        pos_loss = torch.sum(torch.log(torch.sigmoid(torch.sum(u_i * u_jp, dim=1))))

        u_i_neg = self.embed_2(neg_from_)
        u_jp_neg = self.embed_2_context(neg_to_)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-torch.sum(u_i_neg * u_jp_neg, dim=1))))

        return - ( pos_loss + neg_loss )



    def get_embeddings(self, node_idx):

        return torch.cat([self.embed_1(node_idx), self.embed_2(node_idx)], dim=1)





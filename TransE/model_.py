import torch
import torch.nn as nn
from torch import relu, FloatTensor


class TransE(nn.Module):
    def __init__(self, num_entity, num_label, embed_dim, gamma, device):
        super(TransE, self).__init__()
        # 각 entity 별로 embedding space로 투영시키는 작업
        self.embed_entity = nn.Embedding(num_entity, embed_dim)
        # 각 label 별로 embedding space로 투영시키는 작업
        self.embed_label = nn.Embedding(num_label, embed_dim)

        self.gamma = torch.FloatTensor([gamma]).to(device)
        self.embed_dim = embed_dim
        # torch.nn.init.uniform_(self.embed_entity, a=-6/torch.sqrt(torch.tensor(k)), b=6/torch.sqrt(torch.tensor(k)))
        # torch.nn.init.uniform_(self.embed_label, a=-6/torch.sqrt(torch.tensor(k)), b=6/torch.sqrt(torch.tensor(k)))

    def forward(self, batch):
        head, label, tail = batch['head'], batch['label'], batch['tail']
        head_p, tail_p = batch['head_p'], batch['tail_p']

        batch_size = head.size(0)

        # head, tail ( batch size만큼의 head entity, tail entity) 를 embedding space로 transform
        h = self.embed_entity(head)  # (batch_size, embed_dim)
        t = self.embed_entity(tail)  # (batch_size, embed_dim)

        l = self.embed_label(label)  # (batch_size, embed_dim)
        h_p = self.embed_entity(head_p)  # (batch_size, neg_sample, embed_dim)
        t_p = self.embed_entity(tail_p)  # (batch_size, neg_sample, embed_dim)

        pos_d = torch.norm(h + l - t, 2, dim=1)  # (batch_size)
        neg_d1 = torch.norm(h_p + l.unsqueeze(1) - t.unsqueeze(1), 2, dim=2)  # (batch_size, neg_sample)
        neg_d2 = torch.norm((h + l).unsqueeze(1) - t_p, 2, dim=2)  # (batch_size, neg_sample)

        loss = torch.sum(relu(self.gamma + (pos_d.unsqueeze(-1) - neg_d1) + (pos_d.unsqueeze(-1) - neg_d2)), axis=1)

        return torch.sum(loss)

    def distance_l2(self, tensor_h, tensor_l, tensor_t):
        h = self.embed_entity(tensor_h)
        l = self.embed_label(tensor_l)
        t = self.embed_entity(tensor_t)
        return FloatTensor.item(torch.norm(h + l - t, 2))
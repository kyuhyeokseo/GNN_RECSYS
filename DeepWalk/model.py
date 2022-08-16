import torch
from torch import nn


class DeepWalk_model(nn.Module):
    def __init__(self, nodes, config):
        super(DeepWalk_model, self).__init__()

        self.embed_layer = nn.Embedding(len(nodes), config.embed_dim)
        self.nodes = torch.LongTensor(list(nodes)).to(config.device)


    def forward(self, target, window):

        batch_size, _ = target.size()

        embed_target = self.embed_layer(target)
        embed_window = self.embed_layer(window)

        embed_all = self.embed_layer(self.nodes).T.unsqueeze(0).repeat(batch_size, 1, 1)

        score = torch.exp( torch.sum(torch.mul(embed_target, embed_window), dim=2) )
        scale = torch.sum( torch.exp(torch.bmm(embed_target, embed_all)) , dim=2)
        loss = - torch.log(score / scale)

        return torch.mean( torch.mean(loss, dim=1) )




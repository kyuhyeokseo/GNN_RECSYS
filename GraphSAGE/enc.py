import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample,
            base_model=None, cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        if self.aggregator.__class__.__name__ == 'GCNAggregator':
            self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim))
        elif self.aggregator.__class__.__name__ == 'LSTMAggregator':
            self.weight = nn.Parameter(torch.FloatTensor(embed_dim, 3 * self.feat_dim))
        else :
            self.weight = nn.Parameter( torch.FloatTensor(embed_dim, 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes])

        if self.aggregator.__class__.__name__ == 'GCNAggregator':
            combined = neigh_feats
        else :
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))

            combined = torch.cat([self_feats, neigh_feats], dim=1)

        combined = F.relu(self.weight.mm(combined.t()))

        return combined




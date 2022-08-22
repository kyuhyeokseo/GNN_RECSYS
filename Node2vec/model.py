import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data import return_non_adj_list


class Node2vec(nn.Module):
    def __init__(self, graph, nodes, adj_mat, adj_list, config):
        super(Node2vec, self).__init__()

        self.num_nodes = len(nodes)
        self.V = nodes
        self.adj_mat = adj_mat
        self.adj_list = adj_list
        self.config = config
        self.walk_length = config.walk_length
        self.p = config.p
        self.q = config.q

        self.embed_layer = nn.Embedding(len(nodes), config.embed_dim)
        self.non_adj_list = torch.tensor(return_non_adj_list(graph, adj_list, config))

    def node2vecWalk(self, node):

        walk = [node]
        for i in range(self.walk_length-1):

            curr = walk[-1]
            if len(walk)== 1 :
                prev = walk[-1]
            else :
                prev = walk[-2]

            node_candidates, probability = self.get_neighbors(curr, prev)
            node_s = np.random.choice(node_candidates, 1, p=probability)
            walk.append(int(node_s))

        return walk


    def get_neighbors(self, curr, prev):

        if curr == prev :
            neighs = self.adj_list[curr]
            prob_list = [1/len(neighs)]*len(neighs)

        else :
            neighs = self.adj_list[curr]
            probability = []

            for each_neigh in neighs:

                if prev == each_neigh :
                    probability.append(1/self.p)
                elif self.adj_mat[prev, each_neigh]==1:
                    probability.append(1)
                else :
                    probability.append(1/self.q)

            sum_probability = sum(probability)
            prob_list = [ each_prob / sum_probability for each_prob in probability ]

        return neighs, prob_list



    def forward(self, target, window):

        node = target[:,-1]
        non_adj = self.non_adj_list[node]
        batch_size, _ = target.size()

        embed_target = self.embed_layer(target)
        embed_positive = self.embed_layer(window)
        embed_negative = self.embed_layer(non_adj)

        pos = torch.sum(torch.mul(embed_target, embed_positive), dim=2)
        neg = torch.sum(torch.mul(embed_target, embed_negative), dim=2)

        loss = torch.sum( neg - pos )

        return loss
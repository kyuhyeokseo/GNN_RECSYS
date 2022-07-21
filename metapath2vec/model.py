import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from utils import SamplingAliasMethod


class metapath2vec_model(nn.Module):
    def __init__(self, graph, node_type, config):
        super(metapath2vec_model, self).__init__()

        self.graph = graph
        self.node_info = self.get_graph_info(graph)
        self.num_node = graph.number_of_nodes()
        self.X = nn.Embedding(self.num_node, config.embed_dim, device=config.device)
        self.Log_Sig = nn.LogSigmoid()
        self.node_sampler = SamplingAliasMethod(self.node_info[1])
        self.node_type = node_type
        self.config = config


    def get_graph_info(self, graph):
        node_degree = defaultdict(int)
        for edge in graph.edges():
            src = edge[0]
            dst = edge[1]
            node_degree[src] += graph[src][dst]['weight']

        node_idx, node_weight = np.arange(graph.number_of_nodes()), np.zeros(graph.number_of_nodes())
        for n, d in node_degree.items():
            node_weight[n] = np.power(d, 3 / 4)

        return (node_idx, node_weight)

    def forward(self, MP):

        X = self.HeterogeneousSkipGram(self.config.k_neighbor, MP)

        return X


    def MetaPathRandomWalk(self, node):
        l = self.config.walk_len
        MP = []
        MP.append(node)
        type = self.node_type[node]
        metapath_next = self.config.metapath_next

        for i in range(l-1):

            candi_node = np.array(self.graph[node])  # 현재 노드에서 연결되어 있는 노드
            candi_type = np.array(self.node_type)[candi_node]  # 그 노드들의 type
            target_type = metapath_next[type]
            candi_node_filter1 = candi_node[candi_type == target_type]  # 다음 step에서 만족해야하는 type인 노드들

            if len(candi_node_filter1) == 0:
                return MP
            else :
                node = np.random.choice(candi_node_filter1, 1)[0]
                type = self.node_type[node]
                MP.append(node)

        return torch.tensor(MP).long().to(self.config.device)

    def HeterogeneousSkipGram(self, k, MP):
        total_loss = 0
        n = 0

        for i in range(len(MP)):
            v = MP[i]
            for j in range(max(0, i - k), min(i + k + 1, len(MP))):
                if i != j:
                    c_t = MP[j]

                    u_t = torch.tensor(self.node_sampler.return_sample(self.config.neg_sampling_per_pos)).long().to(self.config.device)

                    X_v = self.X(v)
                    X_ct = self.X(c_t)
                    X_ut = self.X(u_t)

                    Loss1 = - self.Log_Sig(torch.sum(torch.mul(X_v, X_ct))) # convert Max -> Min
                    calculate2 = torch.sum(torch.mul(-X_v, X_ut), dim=1)
                    Loss2 = - torch.sum(self.Log_Sig(calculate2))

                    total_loss += (Loss1 + Loss2)
                    n += 1

        if len(MP) :
            return total_loss / n
        else :
            return 0







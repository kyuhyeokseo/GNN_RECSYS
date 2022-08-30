import numpy as np
import torch

from utils import SamplingAliasMethod, get_graph_info


class Line_Dataset:

    def __init__(self, graph, config):

        self.graph = graph
        self.batch_size = config.batch_size
        self.negative_sampling_per_positive = config.negative_sampling_per_positive
        self.config = config

        self.edge_info, self.node_info = get_graph_info(graph)

        self.node_sampler = SamplingAliasMethod(self.node_info[1])
        self.edge_sampler = SamplingAliasMethod(self.edge_info[1])
        self.edge_num = len(self.edge_info[0])
        self.node_num = len(self.node_info[0])

        # After Normalized
        self.edge_w = np.array(self.edge_info[1]) / np.sum(self.edge_info[1])



    def get_batch(self):

        # positive edge sampling
        sampled_edge_idx = self.edge_sampler.return_sample(self.batch_size)
        pos = [self.edge_info[0][each_edge] for each_edge in sampled_edge_idx]
        pos_weight = [self.edge_w[each_edge] for each_edge in sampled_edge_idx]

        # negative edge sampling
        neg = []

        for idx, edge in enumerate(pos):
            start_node = edge[0]
            neg_sampled = self.node_sampler.return_sample(self.negative_sampling_per_positive)
            neg_edges = [(start_node, self.node_info[0][each_neg_node]) for each_neg_node in neg_sampled]
            neg.extend(neg_edges)

        return {'pos': torch.tensor(pos, dtype=torch.long).to(self.config.device),
                'pos_w': torch.tensor(pos_weight, dtype=torch.float).to(self.config.device),
                'neg': torch.tensor(neg, dtype=torch.long).to(self.config.device),
                }
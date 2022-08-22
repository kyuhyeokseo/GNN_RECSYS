import numpy as np

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')

class WalkDataSet(Dataset):
    def __init__(self, graph, model, config):
        super(WalkDataSet, self).__init__()

        self.graph = graph
        self.model = model
        self.walks_per_vertex = config.walks_per_vertex
        self.walk_length = config.walk_length
        self.context_size = config.context_size
        self.nodes = list(graph.nodes())

        self.random_walk_list = []
        for i in self.nodes:
            for j in range(self.walks_per_vertex):
                self.random_walk_list.append(self.model.node2vecWalk(i))


    def __len__(self):
        return len(self.nodes) * int(self.walks_per_vertex) * (self.walk_length - self.context_size)


    def __getitem__(self, idx):

        main_node_list, window_node_list = [], []

        target_random_walk_idx = int(idx/(self.walk_length - self.context_size))
        target_random_walk = self.random_walk_list[target_random_walk_idx]

        context_start_idx = idx % (self.walk_length - self.context_size)
        main_node = target_random_walk[context_start_idx]
        main_node_list.extend([main_node] * self.context_size)
        window = target_random_walk[context_start_idx+1 : context_start_idx + 1 + self.context_size]
        window_node_list.extend(window)

        return {'target':torch.LongTensor(main_node_list), 'window':torch.LongTensor(window_node_list)}




def return_adj(graph):
    adj_list = []
    adj_list.append([])
    for i in range(graph.num_nodes()):
        adj_list.append([])
    A = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for from_, to_ in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        A[from_, to_] += 1
        adj_list[from_].append(to_)
    A = A + np.identity(graph.num_nodes())

    return adj_list, A

def return_non_adj_list(graph, adj_list, config):

    all_node = graph.nodes()
    non_adj_list = []

    for i in range(graph.num_nodes()):

        each_non_adj_list = list(set(all_node) - set(adj_list[i]))
        non_adj = np.random.choice(each_non_adj_list, config.context_size)
        non_adj_list.append(non_adj)

    return non_adj_list

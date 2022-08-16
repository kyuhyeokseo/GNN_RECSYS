import numpy as np
import torch
from torch.utils.data import Dataset


class WalkDataSet(Dataset):
    def __init__(self, graph, walks_per_vertex, walk_length, window_size):
        super(WalkDataSet, self).__init__()

        self.graph = graph
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.window_size = window_size
        self.nodes = list(graph.nodes())


    def __len__(self):
        return len(self.nodes) * int(self.walks_per_vertex)


    def random_walk(self, graph, node, walk_length):

        target_node = node
        walk_list = [target_node]
        for i in range(walk_length):
            next_node_candidates = list(dict(graph[target_node]).keys())
            target_node = np.random.choice(next_node_candidates)
            walk_list.append(target_node)

        return walk_list


    def __getitem__(self, idx):
        node_num = int(idx/self.walks_per_vertex)
        node = self.nodes[node_num]
        walks = self.random_walk(self.graph, node, self.walk_length)

        main_node_list, window_node_list = [], []

        for i, main_node in enumerate(walks):

            left_window = walks[ max(0, i-self.window_size) : i]
            right_window = walks[ i+1 : min(i+1+self.window_size, len(walks))]

            window = left_window + right_window

            main_node_list.extend([main_node] * len(window))
            window_node_list.extend(window)

        return {'target':torch.LongTensor(main_node_list), 'window':torch.LongTensor(window_node_list)}
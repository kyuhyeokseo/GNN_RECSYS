import numpy as np
import pandas as pd
import numpy.random as npr

from collections import defaultdict
import networkx as nx

dd = defaultdict(int)
for _ in range(2000):
    dd[np.random.randint(4)] += 1


def make_graph_data(file_name, weighted, num_type_node):
    data = pd.read_pickle(file_name)
    graph = nx.Graph(data)

    if weighted:
        for edge in graph.edges():
            src = edge[0]
            dst = edge[1]
            graph[src][dst]['weight'] = np.random.randint(0, 100)
    else:
        for edge in graph.edges():
            src = edge[0]
            dst = edge[1]
            graph[src][dst]['weight'] = 1

    node_type = []
    if num_type_node:
        for node in graph.nodes():
            node_type.append(np.random.randint(num_type_node))
        return graph, node_type

    return graph


class SamplingAliasMethod:

    def __init__(self, probs):
        self.probs = np.array(probs) / np.sum(probs)
        self.K = len(probs)
        self.prob_arr = np.zeros(self.K)
        self.alias_arr = np.zeros(self.K, dtype=np.int32)
        self.create_alias_table_(probs)

    def create_alias_table_(self, probs):
        small = []
        large = []

        for idx, prob in enumerate(self.probs):
            self.prob_arr[idx] = prob * self.K
            if self.prob_arr[idx] < 1:
                small.append(idx)
            else:
                large.append(idx)

        while small and large:
            s = small.pop()
            l = large.pop()
            self.alias_arr[s] = l
            self.prob_arr[l] = self.prob_arr[l] - (1 - self.prob_arr[s])

            if self.prob_arr[l] < 1:
                small.append(l)
            else:
                large.append(l)

    def sampling(self):
        random_idx = int(np.floor(np.random.rand() * self.K))
        threshold = self.prob_arr[random_idx]
        if np.random.rand() < threshold:
            return random_idx
        else:
            return self.alias_arr[random_idx]

    def return_sample(self, n_sample):
        return [self.sampling() for _ in range(n_sample)]



def get_graph_info(graph):
        edge_idx, edge_weight = [], []
        node_degree = defaultdict(int)

        for each_edge in graph.edges():
            from_ = each_edge[0]
            to_ = each_edge[1]

            edge_idx.append(each_edge)
            edge_weight.append(graph[from_][to_]['weight'])

            node_degree[from_] += graph[from_][to_]['weight']

        node_idx = np.arange(graph.number_of_nodes())
        node_weight = np.zeros(graph.number_of_nodes())

        for idx, degree in node_degree.items():
            node_weight[idx] = np.power(degree, 3 / 4)

        return (edge_idx, edge_weight), (node_idx, node_weight)


def get_params_group(model, type):
    if type == 'first':
        optimizer_parameters = [{'params': [p for n,p in model.named_parameters() if 'embed_1' in n]}]
    if type == 'second':
        optimizer_parameters = [{'params': [p for n,p in model.named_parameters() if 'embed_2' in n]}]
    return optimizer_parameters
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


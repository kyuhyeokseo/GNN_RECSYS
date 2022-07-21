import numpy as np
import pandas as pd
import os, sys, pickle


import networkx as nx
from copy import deepcopy
from operator import itemgetter

import torch
import torch.nn as nn
from torch import IntTensor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from data import mp2v_Dataset
from model import metapath2vec_model
from utils import make_graph_data, SamplingAliasMethod

from datetime import datetime
import matplotlib.pyplot as plt

import os, random, warnings, math
warnings.filterwarnings('ignore')


class Config:
    learning_rate = 0.005
    weight_decay = 0.01
    epochs = 10
    seed = 1995
    embed_dim = 30
    k_neighbor = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    neg_sampling_per_pos = 5

    num_type_node = 2
    metapath = [0,1,0]
    metapath_next = [1,0]
    walk_len = 16
    metapath_walk = metapath[:-1] * walk_len # 그냥 넉넉히 잡은 것임
    num_walk_per_node = 1 # num_walk_per_node 를 늘릴 수록 loss가 팍 높아지는 시점이 빨라짐 => random walk를 많이 학습할 수록 loss가 높아진다 => gradient exploding?

config = Config()

graph, node_type = make_graph_data('ind.citeseer.graph', weighted=True, num_type_node=config.num_type_node)
config.node_type = np.array(node_type)
# graph : 3327 nodes, 4676 edges, node_type length : 3327 ( 0 or 1 )


dataset = mp2v_Dataset(graph.number_of_nodes())
dataloader = DataLoader(dataset, config.batch_size, drop_last=False, shuffle=True)

model = metapath2vec_model(graph, node_type, config)

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

for epoch in range( config.epochs ):

    loss_list = []
    # Train ON
    model.train()

    for i_1 in range(config.num_walk_per_node):

        for batch_data in dataloader:
            node_list = batch_data['node']
            for node in node_list:

                node = int(node.item())
                MP = model.MetaPathRandomWalk(node)
                if len(MP) > 2:
                    k = config.k_neighbor

                    optimizer.zero_grad()

                    loss = model(MP)
                    loss.backward()

                    optimizer.step()


        loss_list.append(loss.item())

    print(f'EPOCH {epoch+1} : TRAINING Loss {np.mean(loss_list)}')


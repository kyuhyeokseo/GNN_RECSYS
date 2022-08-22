import numpy as np
import torch
import torch.nn.functional as F

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dgl.data import CoraGraphDataset
from data import return_adj, WalkDataSet
from model import Node2vec


class Config():
    learning_rate = 0.01
    weight_decay = 0.01
    epochs = 100
    batch_size = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    walk_length = 7
    walks_per_vertex = 10
    context_size = 3

    p = 0.5
    q = 0.5

    embed_dim = 32

config = Config()

dataset = CoraGraphDataset()
graph = dataset[0]

config.input_dim = graph.ndata['feat'].shape[1]
config.output_dim = graph.ndata['label'].unique().shape[0]

V = graph.nodes()
adj_list, adj_mat = return_adj(graph)
model = Node2vec(graph, V, adj_mat, adj_list, config).to(config.device)

dataset = WalkDataSet(graph, model, config)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay= config.weight_decay)

for epoch in range(config.epochs):

    loss_list = []

    # TRAIN ON
    model.train()

    loss = 0
    optimizer.zero_grad()

    for batch_data in dataloader:

        target, window = batch_data['target'].to(config.device), batch_data['window'].to(config.device)
        loss = loss + model(target, window)

    loss_list.append(loss)

    loss.backward()
    optimizer.step()

    print(f'EPOCH {epoch + 1} : TRAINING LOSS {loss_list[-1]:.4f}')



















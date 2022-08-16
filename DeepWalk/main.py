import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data import WalkDataSet
from model import DeepWalk_model
from utils import make_graph_data

import warnings
warnings.filterwarnings('ignore')


class Config:
    learning_rate = 0.001
    weight_decay = 0.001
    epochs = 30
    seed = 1995

    window_size = 4
    embed_dim = 30
    walks_per_vertex = 10
    walk_length = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 128
    num_type_node = 2



config = Config()

graph, node_type = make_graph_data('ind.citeseer.graph', weighted=True, num_type_node=config.num_type_node)
config.node_type = np.array(node_type)
# graph : 3327 nodes, 4676 edges,
# node_type length : 3327 ( 0 OR 1 )


dataset = WalkDataSet(graph, config.walks_per_vertex, config.walk_length, config.window_size)
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)

model = DeepWalk_model(graph.nodes(), config)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay= config.weight_decay)

for epoch in range(config.epochs):

    loss_list = []

    # TRAIN ON
    model.train()
    for batch_data in dataloader:

        optimizer.zero_grad()
        target, window = batch_data['target'].to(config.device), batch_data['window'].to(config.device)

        loss = model(target, window)

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    print(f'EPOCH {epoch + 1} : TRAINING LOSS {np.mean(loss_list):.4f}')















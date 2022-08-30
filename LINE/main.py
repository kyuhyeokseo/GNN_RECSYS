import warnings
from collections import defaultdict

import numpy as np
import torch
from torch import optim

from data import Line_Dataset
from model import LINE
from utils import make_graph_data, get_params_group

warnings.filterwarnings('ignore')

graph = make_graph_data('ind.citeseer.graph', weighted=True, num_type_node=False)
# graph : 3327 nodes, 4676 edges

class Config:
    learning_rate_1 = 0.001
    learning_rate_2 = 0.001
    epoch_1 = 200
    epoch_2 = 30
    epochs = max(epoch_1, epoch_2)
    weight_decay = 0.001

    embed_dim = 30
    negative_sampling_per_positive = 5
    batch_size = 128

    order = 'all'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

config.num_batch_per_epoch = graph.number_of_edges() // config.batch_size

dataset = Line_Dataset(graph, config)
model = LINE( n_node = graph.number_of_nodes(), embed_dim = config.embed_dim )
model = model.to(config.device)

optimizer_1 = optim.SGD(get_params_group(model, 'first'), lr=config.learning_rate_1, momentum=0.9, weight_decay = config.weight_decay)
optimizer_2 = optim.SGD(get_params_group(model, 'second'), lr=config.learning_rate_2, momentum=0.9, weight_decay = config.weight_decay)


history_1, history_2 = defaultdict(list), defaultdict(list)

for epoch in range(config.epochs):

    model.train()
    loss_list_1, loss_list_2 = [], []

    for iteration in range(config.num_batch_per_epoch):

        batch_data = dataset.get_batch()

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss = model(batch_data)

        if config.epoch_1 > epoch:

            loss_1 = loss['first']
            loss_1.backward()
            optimizer_1.step()
            loss_list_1.append(loss_1.item())

        if config.epoch_2 > epoch:

            loss_2 = loss['second']
            loss_list_2.append(loss_2.item())
            loss_2.backward()
            optimizer_2.step()


    display_f = f' --- END --- ' if np.isnan(np.mean(loss_list_1)) else np.round(np.mean(loss_list_1), 6)
    display_s = f' --- END --- ' if np.isnan(np.mean(loss_list_2)) else np.round(np.mean(loss_list_2), 6)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f'EPOCH {epoch + 1} TRAIN FIRST LOSS : {display_f}, TRAIN SECOND LOSS : {display_s}')
































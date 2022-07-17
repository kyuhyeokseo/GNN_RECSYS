
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import warnings

from model_ import GCN

warnings.filterwarnings('ignore')

from dgl.data import CoraGraphDataset

from data_ import GCNDataset, Ahat_mat
import warnings
warnings.filterwarnings('ignore')

class Config:
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 32
    epochs = 200
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 919

config = Config()
dataset = CoraGraphDataset()
graph = dataset[0]
config.batch_size = graph.num_nodes()
config.input_dim = graph.ndata['feat'].shape[1]
config.output_dim = graph.ndata['label'].unique().shape[0]

dataset = GCNDataset(graph, True)
dataset_test = GCNDataset(graph, False)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

A = Ahat_mat(graph, config)

model = GCN(config)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# To use Cross Entropy Loss
loss_fn = nn.CrossEntropyLoss()

for epoch in range(config.epochs):

    # Train ON
    model.train()
    train_acc_list = []
    test_acc_list = []

    for batch_data in dataloader:
        # gradient initializing
        optimizer.zero_grad()

        data, label = model(batch_data, A)
        acc_train = torch.sum(label == torch.argmax(data, axis=1)) / len(label)
        train_Loss = loss_fn(data, label)
        train_Loss.backward()
        optimizer.step()



    train_acc_list.append(acc_train)

    for batch_data in dataloader_test:

        data, label = model(batch_data, A)
        test_Loss = loss_fn(data, label)

        acc_test = torch.sum(label == torch.argmax(data, axis=1)) / len(label)

    test_acc_list.append(acc_test)

    print(f'EPOCH {epoch + 1} : TRAIN ACC : {train_acc_list[-1]:.4f}, TEST ACC : {test_acc_list[-1]:.4f}')


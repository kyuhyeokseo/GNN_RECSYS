import torch
import torch.nn.functional as F

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dgl.data import CoraGraphDataset
from data import VGAEDataset,Ahat_mat

from model import VGAE

class Config():
    learning_rate = 0.01
    epochs = 100
    batch_size = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    latent_size = 16
    hidden_dim = 32

config = Config()

dataset = CoraGraphDataset()
graph = dataset[0]
config.batch_size = graph.num_nodes()
config.input_dim = graph.ndata['feat'].shape[1]
config.output_dim = graph.ndata['label'].unique().shape[0]

dataset = VGAEDataset(graph, True)
dataset_test = VGAEDataset(graph, False)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

A_til, A = Ahat_mat(graph, config)

model = VGAE(config.input_dim, config.hidden_dim, config.latent_size, A_til).to(config.device)
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)


def loss_function(recon_A, A, mu, logvar):
    BCE = F.binary_cross_entropy(recon_A, torch.tensor(A).float(), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


train_acc_list = []
test_acc_list = []


for epoch in range( config.epochs ):

    # TRAIN ON
    model.train()

    total_loss = 0
    total_loss_test = 0

    for batch_data in dataloader:
        # gradient initializing
        feats = batch_data['x']
        x = batch_data['y']
        optimizer.zero_grad()

        A_pred, mu, log_var = model(feats)

        BCE, KLD = loss_function(A_pred, A, mu, log_var)
        loss = BCE + KLD
        loss.backward()

        total_loss += loss.item()

        optimizer.step()

    avg_loss = total_loss / len(dataloader.dataset)

    for batch_data in dataloader_test:
        # gradient initializing
        feats = batch_data['x']
        x = batch_data['y']
        optimizer.zero_grad()

        A_pred, mu, log_var = model(feats)

        BCE, KLD = loss_function(A_pred, A, mu, log_var)
        loss = BCE + KLD

        total_loss_test += loss.item()

    avg_loss_test = total_loss_test / len(dataloader_test.dataset)


    print(f'EPOCH {epoch + 1} : TRAIN ACC : {avg_loss:.4f}, TEST ACC : {avg_loss_test:.4f}')

import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import VAE

class Config():
    learning_rate = 0.01
    epochs = 30
    batch_size = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    latent_size = 2
    hidden_dim1 = 512
    hidden_dim2 =256

config = Config()

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# Data Loader
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

model = VAE(28*28, config.hidden_dim1, config.hidden_dim2, config.latent_size).to(config.device)
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


for epoch in range( config.epochs ):

    # TRAIN ON
    model.train()
    loss_sum = 0
    loss_sum_test = 0

    # TRAIN
    for batch_idx, (data, _) in enumerate(train_loader):

        optimizer.zero_grad()

        reconstruct_x, mu, log_var = model(data)

        BCE, KLD = loss_function(reconstruct_x, data, mu, log_var)
        loss = BCE + KLD

        loss.backward()

        loss_sum += loss.item()

        optimizer.step()

    avg_loss = loss_sum / len(train_loader.dataset)

    # TEST
    model.eval()
    for batch_t_idx, (data_t, _) in enumerate(test_loader):

        reconstruct_x, mu, log_var = model(data_t)

        BCE, KLD = loss_function(reconstruct_x, data_t, mu, log_var)
        loss = BCE + KLD

        loss_sum_test += loss.item()

    avg_loss_test = loss_sum_test / len(test_loader.dataset)

    print(f'EPOCH {epoch + 1} : TRAINING RMSE LOSS {avg_loss:.4f}, TEST RMSE LOSS {avg_loss_test:.4f}')






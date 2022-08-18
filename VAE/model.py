import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim1, hidden_dim2, z_dim):
        super(VAE, self).__init__()

        # ENCODER
        self.fc1 = nn.Linear(x_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, z_dim)
        self.fc32 = nn.Linear(hidden_dim2, z_dim)

        # DECODER
        self.fc4 = nn.Linear(z_dim, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc6 = nn.Linear(hidden_dim1, x_dim)

    def encoder(self, x):

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        return self.fc31(h), self.fc32(h)


    def sampling(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)


    def decoder(self, z):

        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))

        return torch.sigmoid(self.fc6(h))


    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
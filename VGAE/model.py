import torch
from torch import nn
import torch.nn.functional as F

class VGAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, z_dim, adj):
        super(VGAE, self).__init__()

        self.input_dim = x_dim

        # ENCODER
        self.layer1 = nn.Linear(x_dim, hidden_dim)
        self.layer_mu = nn.Linear(hidden_dim, z_dim)
        self.layer_logstddev = nn.Linear(hidden_dim, z_dim)
        self.adj = adj


    def encoder(self, x):

        x = self.layer1(x)
        x = torch.mm(self.adj, x)
        x = F.relu(x)

        mu = torch.mm(self.adj, self.layer_mu(x))
        logstddev = torch.mm(self.adj, self.layer_logstddev(x))

        return mu, logstddev


    def sampling(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)


    def decoder(self, Z):

        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))

        return A_pred


    def forward(self, x):
        mu, log_var = self.encoder(x)
        Z = self.sampling(mu, log_var)
        return self.decoder(Z), mu, log_var
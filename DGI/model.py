from torch import nn
from layer import GCN, ReadOut, Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_hidden ):
        super(DGI, self).__init__()

        self.GCN = GCN(n_in, n_hidden)
        self.ReadOut = ReadOut()
        self.sigmoid = nn.Sigmoid()
        self.Discriminator = Discriminator(n_hidden)


    def forward(self, X, X_prime, adj,  ):

        H = self.GCN(X, adj)
        H_prime = self.GCN(X_prime, adj)

        s = self.ReadOut(H).expand_as(H)

        output = self.Discriminator(H, H_prime, s)

        return output





        return
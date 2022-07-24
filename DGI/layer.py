import torch
from torch import nn

class GCN(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCN, self).__init__()

        self.W = nn.Linear(n_in, n_out, bias = False)
        torch.nn.init.xavier_uniform_(self.W.weight.data)
        self.act = nn.PReLU()

    def forward(self, seq, adj):

        output = torch.spmm(adj, seq)
        output = self.W(output)
        output = self.act(output)

        return output

class ReadOut(nn.Module):
    def __init__(self):
        super(ReadOut, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, H):
        return self.sigmoid(torch.mean(H, dim=0)).unsqueeze(0)


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()

        self.f_k = nn.Bilinear(n_hidden, n_hidden, 1)

        torch.nn.init.xavier_uniform_(self.f_k.weight.data)


    def forward(self, H, H_prime, s):

        score = self.f_k(H, s).squeeze().unsqueeze(0)
        score_prime = self.f_k(H_prime, s).squeeze().unsqueeze(0)

        output = torch.cat((score, score_prime),1)

        return output





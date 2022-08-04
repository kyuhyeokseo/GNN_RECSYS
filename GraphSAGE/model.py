import random

import numpy as np
import torch
from torch import nn
from torch.nn import init


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())




class UnsupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, degree_list, adj_list):
        super(UnsupervisedGraphSage, self).__init__()
        self.enc = enc
        self.degree_list = degree_list
        self.adj_list = adj_list
        wt = np.power(degree_list, 0.75)
        wt = wt / wt.sum()
        self.weights = torch.FloatTensor(wt)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))

    def negative_sample(self, number_of_neg_sample):
        return torch.multinomial(self.weights, number_of_neg_sample, replacement=True)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds

    def affinity(self, input_1, input_2):
        output_1 = torch.nn.functional.normalize(self.forward(input_1))
        output_2 = torch.nn.functional.normalize(self.forward(input_2))
        aff = torch.sum((output_1 * output_2), dim=1)
        return output_1, aff

    def neg_affinity(self, output_1, neg_samples):
        neg_output = torch.nn.functional.normalize(self.forward(neg_samples))
        neg_aff = torch.mm(output_1.t(), neg_output)
        return neg_aff

    def score(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()




class f1_weight_model(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(f1_weight_model, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        self.xent = nn.CrossEntropyLoss()

    def forward(self, embeds):
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

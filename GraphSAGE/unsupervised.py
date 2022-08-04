import math
import random
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, FloatTensor
from torch.autograd import Variable

from agg import MeanAggregator, meanPoolAggregator, GCNAggregator, LSTMAggregator
from enc import Encoder
from model import SupervisedGraphSage, UnsupervisedGraphSage, f1_weight_model
from utils import load_cora

warnings.filterwarnings('ignore')
"""
Unsupervised GraphSAGE MODEL
"""

print("------ Unsupervised GraphSAGE MODEL ------")

class Config():
    epochs = 100
    num_sample = 2
    batch_size = 256

num_nodes = 2708
# features : 1433, labels : 7, adj_list : set
feat_data, labels, adj_lists = load_cora()

config = Config()

features = nn.Embedding(2708, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
# features.cuda()


rand_indices = np.random.permutation(num_nodes)
test = rand_indices[:1000]
val = rand_indices[1000:1500]
train = list(rand_indices[1500:]) # 1208 nodes

train_degree_list = [len(adj_lists[node]) for node in train]
train_edges = [(row, node) for row in train for node in adj_lists[row] if node in train]
val_edges = [(row, node) for row in val for node in adj_lists[row] if node in val]
test_edges = [(row, node) for row in test for node in adj_lists[row] if node in test]

number_of_neg_sample = 5
pos_train_list = [ random.choice(list(adj_lists[node])) for node in train ]
neg_train_list = [ random.sample( list(set(train).difference(adj_lists[node])) , number_of_neg_sample ) for node in train ]

pos_val_list = [ random.choice(list(adj_lists[node])) for node in val ]
neg_val_list = [ random.sample( list(set(val).difference(adj_lists[node])) , number_of_neg_sample ) for node in val ]

pos_test_list = [ random.choice(list(adj_lists[node])) for node in test ]
neg_test_list = [ random.sample( list(set(test).difference(adj_lists[node])) , number_of_neg_sample ) for node in test ]


# Model
agg1 = MeanAggregator(features, feat_dim = 1433, num_sample= config.num_sample, cuda=True)
enc1 = Encoder(features, 1433, 32, adj_lists, agg1, num_sample= config.num_sample, cuda=False)
agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), feat_dim = enc1.embed_dim, num_sample= config.num_sample, cuda=False)
enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 32, adj_lists, agg2, num_sample= config.num_sample, base_model=enc1, cuda=False)

graphsage = UnsupervisedGraphSage(7, enc2, train_degree_list, adj_lists)

optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)


times = []

rand_permu = np.random.permutation(train)
batch_num = math.ceil(len(rand_permu) / config.batch_size)
batch_list = []
for i in range(batch_num):
    if i == batch_num-1 :
        batch_list.append(rand_permu[i*config.batch_size : ])
    else :
        batch_list.append(rand_permu[i*config.batch_size : (i+1)*config.batch_size])

test_f1_list = []
avg_batch_time_list = []
best = 0

f1_model = f1_weight_model(7, 32)
optimizer_prime = torch.optim.SGD(filter(lambda p: p.requires_grad, f1_model.parameters()), lr=0.1)

train_prime = train

total_loss_past = 0

for epoch in range(config.epochs):

    graphsage.train()
    start_time = time.time()

    total_loss = 0
    optimizer.zero_grad()
    for idx in range(len(train)) :


        node = [train[idx]]
        pos = [pos_train_list[idx]]
        negs = neg_train_list[idx]
        z_u = graphsage(node)
        z_pos = graphsage(pos)
        z_negs = graphsage(negs).t()

        output_u = torch.nn.functional.normalize(z_u)
        output_pos = torch.nn.functional.normalize(z_pos)
        output_negs = torch.nn.functional.normalize(z_negs, dim=1)
        aff = torch.sum((output_u * output_pos), dim=0)

        neg_aff = torch.mm(output_negs, output_u)

        total_loss += - torch.sum(torch.log(torch.sigmoid(aff)))
        total_loss += - number_of_neg_sample * torch.sum((torch.log(torch.sigmoid(-neg_aff))))

    total_loss.backward()
    optimizer.step()
    end_time = time.time()
    times.append(end_time - start_time)

    if epoch>13 and total_loss >= total_loss_past :
        print("------- Done : Learning Embedding Function -------")
        break
    else :
        total_loss_past = total_loss

    print(f'EPOCH : {epoch + 1}, TEST_LOSS : {total_loss : .2f}')

graphsage.eval()

for epoch in range(config.epochs):

    batch_nodes = train_prime[:config.batch_size]
    random.shuffle(train_prime)

    optimizer_prime.zero_grad()
    embeds = graphsage(batch_nodes).data.numpy()
    embeds = torch.from_numpy(embeds)
    loss_prime = f1_model.loss(embeds, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    loss_prime.backward()
    optimizer_prime.step()

    embeds_val = graphsage(val)
    val_output = f1_model(embeds_val)
    val_f1 = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")

        #val_output = graphsage.score(val)
        #test_output = graphsage.score(test)

        #test_f1 = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")

        #if test_f1 > best:
        #    best = test_f1

    print(f'EPOCH : {epoch + 1}, LOSS_prime : {loss_prime.item():.4f}, F1_Score : {val_f1:.4f}')


avg_time = np.mean(times)
test_f1_list.append(best)
avg_batch_time_list.append(avg_time)

print(f'AVG_BATCH_TIME : {avg_time:.4f}')





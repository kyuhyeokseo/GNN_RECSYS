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
from model import SupervisedGraphSage
from utils import load_cora

warnings.filterwarnings('ignore')

class Config():
    epochs = 100
    num_sample = 4
    batch_size = 256

num_nodes = 2708
# features : 1433, labels : 7, adj_list : set
feat_data, labels, adj_lists = load_cora()

config = Config()

features = nn.Embedding(2708, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
# features.cuda()


rand_indices = np.random.permutation(num_nodes)
test = list(rand_indices[:1000])
val = list(rand_indices[1000:1500])
train = list(rand_indices[1500:])


# Model
agg1 = meanPoolAggregator(features, feat_dim = 1433, num_sample= config.num_sample, cuda=True)
enc1 = Encoder(features, 1433, 64, adj_lists, agg1, num_sample= config.num_sample, cuda=False)
agg2 = meanPoolAggregator(lambda nodes : enc1(nodes).t(), feat_dim = enc1.embed_dim,
                          num_sample= config.num_sample, cuda=False)
enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 64, adj_lists, agg2,
               num_sample= config.num_sample, base_model=enc1, cuda=False)

graphsage = SupervisedGraphSage(7, enc2)
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

for epoch in range(config.epochs):

    batch_nodes = train[:config.batch_size]
    random.shuffle(train)

    graphsage.train()

    start_time = time.time()

    optimizer.zero_grad()
    loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    loss.backward()
    optimizer.step()

    end_time = time.time()
    times.append(end_time-start_time)

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    val_f1 = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    test_f1 = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")


    if test_f1>best:
        best = test_f1

    print(loss.item())
    print(f'EPOCH : {epoch+1}, Sup F1_SCORE : {val_f1:.4f}')

avg_time = np.mean(times)
test_f1_list.append(best)
avg_batch_time_list.append(avg_time)

print(f'GraphSAGE-mean : Sup F1_SCORE : {best:.4f}, BATCH_TIME : {avg_time:.4f}')





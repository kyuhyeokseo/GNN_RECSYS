import numpy as np
import pandas as pd
import scipy as scipy

import torch
from scipy.sparse import coo_matrix

from torch import optim, nn
from torch.utils.data import DataLoader

from data_ import BPRDataset
from model_ import BPR_MF

train = np.load('dataset/ml_100k_train.npy')
test = np.load('dataset/ml_100k_test.npy')
#item = pd.read_csv('dataset/movies.csv').iloc[:,5:].values.astype(np.float64)

train = (train >= 4).astype(float)
train = train[0:train.shape[0] // 16, 0:train.shape[1] // 16]
test = (test >= 4).astype(float)
test = test[0:test.shape[0] // 16, 0:test.shape[1] // 16]

class Config():
    learning_rate = 0.001
    weight_decay = 0.001
    embed_dim = 50
    n_user, n_item = train.shape
    epochs = 300
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    neg_sample_size = 4
    lam = 0.001
    batch_size = 512
    margin = 0.5


config = Config()

idx = np.arange(config.n_item)

neg_item = {i :idx[~train[i, :].astype(bool)] for i in range(config.n_user)}

neg_item_test = {i :idx[~test[i, :].astype(bool)] for i in range(config.n_user)}

pos_item_bool = {i :train[i, :].astype(bool) for i in range(config.n_user)}
pos_item_test_bool = {i :test[i, :].astype(bool) for i in range(config.n_user)}

pos_user = train.nonzero()[0]
pos_item = train.nonzero()[1]

pos_user_test = test.nonzero()[0]
pos_item_test = test.nonzero()[1]

dataset = BPRDataset(train, pos_user, pos_item, neg_item, config)
dataset_test = BPRDataset(test, pos_user_test, pos_item_test, neg_item_test, config)

dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size*10, drop_last=False, shuffle=True)


model_MF = BPR_MF(neg_item, config)
model_MF = model_MF.to(config.device)
optimizer = optim.Adam(model_MF.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

for epoch in range( config.epochs ):
    # Train ON
    model_MF.train()

    for batch_data in dataloader:
        u, i, j = batch_data['user'], batch_data['item'], batch_data['neg_item']
        # gradient initializing
        optimizer.zero_grad()
        prediction_i, prediction_j = model_MF(u,i,j)

        loss = (prediction_i - prediction_j).sigmoid().log()

        loss.backward()




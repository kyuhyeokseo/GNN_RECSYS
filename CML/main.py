import numpy as np
import pandas as pd

import torch

from torch import optim
from torch.utils.data import DataLoader


from data_ import CMLDataset
from model_ import CML

train = np.load('dataset/ml_100k_train.npy')
test = np.load('dataset/ml_100k_test.npy')
item = pd.read_csv('dataset/movies.csv').iloc[:,5:].values.astype(np.float64)

train = (train >= 4).astype(float)
test = (test >= 4).astype(float)

class Config():
    learning_rate = 0.001
    weight_decay = 0.001
    neg_sample_size = 10
    batch_size = 512
    margin = 0.5
    embed_dim = 100
    n_user, n_item = train.shape
    epochs = 300
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    rankK = 100

config = Config()

idx = np.arange(config.n_item)

neg_item = {i :idx[~train[i, :].astype(bool)] for i in range(config.n_user)}
neg_item_test = {i :idx[~test[i, :].astype(bool)] for i in range(config.n_user)}

pos_item_bool = {i :train[i, :].astype(bool) for i in range(config.n_user)}
pos_item_test_bool = {i :test[i, :].astype(bool) for i in range(config.n_user)}

pos_user = train.nonzero()[0]
pos_item = train.nonzero()[1]

dataset = CMLDataset(train, item, config.neg_sample_size, neg_item)
dataset_test = CMLDataset(test, item, config.neg_sample_size, neg_item)
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size*10, drop_last=False, shuffle=False)

model = CML(item.shape[1], neg_item, config)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

for epoch in range( config.epochs ):
    # Train ON
    model.train()
    Loss_sum = 0
    #print(f'PROCESSING : EPOCH : {epoch+1}')

    for batch_data in dataloader:
        # gradient initializing
        optimizer.zero_grad()

        batch_data = {k: v.to(config.device) for k, v in batch_data.items()}

        loss = model(batch_data)
        Loss_sum += loss.item()

        loss.backward()
        optimizer.step()




    Loss_past = loss.item()
    recall = []
    recall_test = []

    for u in torch.arange(config.n_user).to(config.device):
        U_i = model.embed_u(u)
        scores = torch.sum( (U_i - model.embed_v.weight.data) ** 2, axis=1).detach().cpu().numpy()
        # score 에 따른 rank 매기는 작업 후 top-K 에 맞추어 mask 추가
        rank = scores.argsort().argsort()
        topK_mask = (rank <= config.rankK)

        # user u 에 대해 pos_item 뽑아내기
        pos_item_user = pos_item_bool[u.item()]
        pos_item_user_test = pos_item_test_bool[u.item()]

        # Positive item 중 top-K item 비율 뽑기
        if pos_item_user.sum() > 0:
            recall.append((topK_mask * pos_item_user).sum() / pos_item_user.sum())
        if pos_item_user_test.sum() >= 5:
            recall_test.append((topK_mask * pos_item_user_test).sum() / pos_item_user_test.sum())

    recall, recall_test = np.mean(recall), np.mean(recall_test)

    if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == config.epochs:
        print(f'EPOCH {epoch + 1} : train loss {Loss_sum : .0f}, train recall@{config.rankK} {recall: .4f}, valid recall@{config.rankK} {recall_test: .4f}')




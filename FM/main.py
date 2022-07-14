

import numpy as np
import torch as torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 데이터 로드
from torch import optim, nn
from torch.utils.data import DataLoader

from data_ import FMDataset
from model_ import FM

scaler = MinMaxScaler()
file = load_breast_cancer()
X, Y = file['data'], file['target']
X = scaler.fit_transform(X)

class Config():
    learning_rate = 0.001
    weight_decay = 0.001

    batch_size = 64
    size = 20

    epochs = 30
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

config = Config()

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=919)

dataset = FMDataset( x_train, y_train )
dataset_test = FMDataset( x_test, y_test )
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size*10, drop_last=False, shuffle=False)

model = FM(x_train, y_train, config)
model = model.to(config.device)

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

for epoch in range( config.epochs ):
    # Train ON
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    train_Loss_sum = 0
    test_Loss_sum = 0

    for batch_data in dataloader:
        # gradient initializing
        optimizer.zero_grad()

        batch_data = {k: v.to(config.device) for k, v in batch_data.items()}

        real = batch_data['value']

        pred = model(batch_data)

        train_Loss = loss_fn(pred, real)
        train_Loss.backward()
        optimizer.step()

        train_Loss_sum += train_Loss.item()

    train_Loss_avg = train_Loss_sum / len(dataloader)

    for batch_data in dataloader_test:

        pred = model(batch_data)
        real = batch_data['value']

        test_Loss = loss_fn(pred, real)

        test_Loss_sum += test_Loss.item()

    test_Loss_avg = test_Loss_sum / len(dataloader_test)

    print(f'EPOCH {epoch + 1} : TRAIN AVG LogLoss : {train_Loss_avg:.6f}, TEST AVG LogLoss : {test_Loss_avg:.6f}')
















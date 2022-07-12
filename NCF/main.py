import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from MLP_data_ import MLP_Dataset, MLP_Dataset_test
from MLP_model_ import MLP
from NCF_model_ import NCF
from data_ import Dataset, Dataset_test


class Config():
    learning_rate = 0.0001
    weight_decay = 0.01
    early_stopping_round = 0
    epochs = 20
    seed = 919
    embed_dim = 16
    hidden_dim = [32,16,8]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128

    pretrained = True

    pretrained_path = f'pretrained'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main_():

    train = np.load('dataset/ml_100k_train.npy')
    test = np.load('dataset/ml_100k_test.npy')

    # make implicit data
    train = (train >= 4).astype(float)
    test = (test >= 4).astype(float)

    config = Config()
    seed_everything(config.seed)

    dataset = Dataset(train, neg_data_per_pos_data=4)
    dataset_test = Dataset_test(test)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, drop_last=False, shuffle=False)

    model = NCF(train.shape[0], train.shape[1], config)
    model = model.from_pretrained(config.pretrained_path, model)

    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # To use Binary Cross Entropy Loss
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):

        # Train ON
        model.train()
        train_Loss_sum = 0
        test_Loss_sum = 0

        for batch_data in dataloader:

            # gradient initializing
            optimizer.zero_grad()

            user = batch_data[0].to(config.device, dtype=torch.long)
            item = batch_data[1].to(config.device, dtype=torch.long)
            real = batch_data[2].to(config.device, dtype=torch.float)

            pred = model(user = user, item = item)

            train_Loss = loss_fn(pred, real.unsqueeze(-1))
            train_Loss.backward()
            optimizer.step()

            train_Loss_sum += train_Loss.item()

        train_Loss_avg = train_Loss_sum / len(dataloader)

        for batch_data in dataloader_test:

            user = batch_data[0].to(config.device, dtype=torch.long)
            item = batch_data[1].to(config.device, dtype=torch.long)
            real = batch_data[2].to(config.device, dtype=torch.float)

            pred = model(user = user, item = item)
            test_Loss = loss_fn(pred, real.unsqueeze(-1))

            test_Loss_sum += test_Loss.item()

        test_Loss_avg = test_Loss_sum / len(dataloader_test)

        print(f'EPOCH {epoch + 1} : TRAIN AVG LogLoss : {train_Loss_avg:.6f}, TEST AVG LogLoss : {test_Loss_avg:.6f}')






main_()


import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from GMF_data_ import GMF_Dataset, GMF_Dataset_test
from GMF_model_ import GMF


class Config():
    learning_rate = 0.0001
    weight_decay = 0.01
    early_stopping_round = 0
    epochs = 20
    seed = 919
    embed_dim = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def GMF_main_():

    train = np.load('dataset/ml_100k_train.npy')
    test = np.load('dataset/ml_100k_test.npy')

    # make implicit data
    train = (train >= 4).astype(float)
    test = (test >= 4).astype(float)

    config = Config()
    seed_everything(config.seed)

    dataset = GMF_Dataset(train, neg_data_per_pos_data=4)
    dataset_test = GMF_Dataset_test(test)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config.batch_size, drop_last=False, shuffle=False)

    model = GMF(train.shape[0], train.shape[1], config)
    model = model.to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # To use Binary Cross Entropy Loss
    loss_fn = nn.BCEWithLogitsLoss()

    best_test_loss = 10

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

            pred = model(user = user, item = item, binary = real)
            train_Loss = loss_fn(pred, real.unsqueeze(-1))
            train_Loss.backward()
            optimizer.step()

            train_Loss_sum += train_Loss.item()

        train_Loss_avg = train_Loss_sum / len(dataloader)

        for batch_data in dataloader_test:

            user = batch_data[0].to(config.device, dtype=torch.long)
            item = batch_data[1].to(config.device, dtype=torch.long)
            real = batch_data[2].to(config.device, dtype=torch.float)

            pred = model(user = user, item = item, binary = real)
            test_Loss = loss_fn(pred, real.unsqueeze(-1))

            test_Loss_sum += test_Loss.item()

        test_Loss_avg = test_Loss_sum / len(dataloader_test)

        if best_test_loss > test_Loss_avg :
            # update 될수록 성능좋아진 모델 저장
            best_test_loss = test_Loss_avg
            torch.save(model.state_dict(), f'pretrained/pretrained_GMF.pth')
            print(f'EPOCH {epoch + 1} : TRAIN AVG LogLoss : {train_Loss_avg:.6f}, TEST AVG LogLoss : {test_Loss_avg:.6f} ... Best Test Loss UPDATE : ---- GMF Model Saving ----')
        else :
            print(f'EPOCH {epoch + 1} : TRAIN AVG LogLoss : {train_Loss_avg:.6f}, TEST AVG LogLoss : {test_Loss_avg:.6f}')






GMF_main_()
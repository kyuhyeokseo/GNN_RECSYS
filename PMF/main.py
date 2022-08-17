import numpy as np
from numpy import sqrt
from scipy.sparse import coo_matrix

from torch import optim
from model import PMF


class Config:
    learning_rate = 0.05
    early_stopping_round = 0
    epochs = 30
    seed = 1995
    embed_dim = 30
    batch_size = 1024
    sigma = 0.01
    sigma_u = 0.1

config = Config()

lambda_ = (config.sigma / config.sigma_u) ** 2

train = np.load('ml_100k_train.npy')
test = np.load('ml_100k_test.npy')

user_num = train.shape[0]
item_num = train.shape[1]
max_rate = 5

model = PMF(train, user_num, item_num, max_rate, config)

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = lambda_)

cx_train = coo_matrix(train)
cx_test = coo_matrix(test)

for epoch in range( config.epochs ):

    # Train ON
    model.train()
    loss = 0
    loss_test = 0
    optimizer.zero_grad()

    for u, i, v in zip(cx_train.row, cx_train.col, cx_train.data):

        loss_each = model(u,i,v)
        loss = loss + loss_each

    loss.backward()
    optimizer.step()

    avg_loss = sqrt(loss.item()/len(cx_train.row))

    for u_test, i_test, v_test in zip(cx_test.row, cx_test.col, cx_test.data):

        loss_each_test = model(u_test,i_test,v_test)
        loss_test = loss_test + loss_each_test

    avg_loss_test = sqrt(loss_test.item() / len(cx_test.row))

    print(f'EPOCH {epoch + 1} : TRAINING RMSE LOSS {avg_loss:.4f}, TEST RMSE LOSS {avg_loss_test:.4f}')




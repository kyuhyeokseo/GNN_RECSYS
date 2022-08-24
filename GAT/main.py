import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import nll_loss

from model import GAT
from utils import load_data, accuracy

import warnings
warnings.filterwarnings('ignore')


class Config:
    learning_rate = 0.005
    weight_decay = 5e-4
    epochs = 30
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 919

    nhid = 4
    dropout = 0.6
    alpha = 0.2
    nheads = 4


config = Config()

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GAT( nfeat = features.shape[1], nhid = config.nhid, nclass = int(labels.max()) + 1,
             dropout = config.dropout, alpha = config.alpha, nheads = config.nheads )
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)

loss_fn = nll_loss

train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(config.epochs):

    # TRAIN ON
    model.train()

    optimizer.zero_grad()
    output = model(features, adj)
    print("main.py : line49")
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    print("main.py : line51")
    loss_train.backward()
    print("main.py : line53")
    optimizer.step()
    print("main.py : line55")

    train_loss_list.append(loss_train.item())

    # TRAIN, TEST ACCURACY
    accuracy_train = accuracy(output[idx_train], labels[idx_train])
    accuracy_test = accuracy(output[idx_test], labels[idx_test])

    train_acc_list.append(accuracy_train)
    test_acc_list.append(accuracy_test)

    print(f'EPOCH {epoch + 1} : TRAIN LOSS : {train_loss_list[-1]:.4f}, '
          f'TRAIN ACC : {train_acc_list[-1]:.4f}, TEST ACC : {test_acc_list[-1]:.4f}')





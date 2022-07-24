import torch
from torch import nn, optim

from model import DGI
from process import load_data, get_A_mat, make_X_prime

class Config:
    lr = 0.001
    weight_decay = 0.0
    hidden_dim = 128
    epochs = 500
    patience = 10
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

config = Config()

# 2708 Nodes, 10556 Edges, 1433 Feats, 7 classes, 140 tr.sample, 500 val.sample, 1000 test.samle
graph, adj, X, Y, train_mask, test_mask = load_data()

config.input_dim = X.shape[1]
label = torch.cat((torch.ones(X.size(0)), torch.zeros(X.size(0))))

adj = get_A_mat(graph, config)
X = X.to(config.device)
label = label.to(config.device)

model = DGI(config.input_dim, config.hidden_dim).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


loss_fn = nn.BCEWithLogitsLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(config.epochs):

    X_prime = make_X_prime(X).to(config.device)

    model.train()
    optimizer.zero_grad()

    pred = model(X, X_prime, adj)

    loss = loss_fn(pred, label.unsqueeze(0))

    print(f'EPOCH {epoch + 1}: Train Loss {loss.item():.5f}')

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == config.patience:
        print('Early stopping!')
        break

    loss.backward()
    optimizer.step()

print('Loading {}th epoch'.format(best_t))







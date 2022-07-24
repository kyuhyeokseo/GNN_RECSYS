from dgl.data import CoraGraphDataset
import numpy as np
import torch

def load_data():
    graph = CoraGraphDataset()[0]
    edges = graph.edges()
    adj = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for src, dst in zip(edges[0].numpy(), edges[1].numpy()):
        adj[src, dst] += 1

    feat, label, train_mask, test_mask = graph.ndata['feat'], graph.ndata['label'], graph.ndata['train_mask'], graph.ndata['test_mask']
    return graph, torch.tensor(adj).float(), feat, label.numpy(), train_mask.numpy(), test_mask.numpy()


def get_A_mat(graph, config):
    A = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for src, dst in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        A[src, dst] += 1
    A = A + np.identity(graph.num_nodes())
    D = np.sum(A, axis=1)
    D = np.diag(np.power(D, -0.5))
    Ahat = np.dot(D, A).dot(D)
    return torch.tensor(Ahat).float().to(config.device)

def make_X_prime(X):
    num_nodes = X.shape[0]
    shuffled_idx = torch.randperm(num_nodes)
    X_prime = X[shuffled_idx, :]
    return X_prime



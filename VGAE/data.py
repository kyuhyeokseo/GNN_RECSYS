import numpy as np

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')

class VGAEDataset(Dataset):
    def __init__(self, graph, is_train):
        super(VGAEDataset, self).__init__()
        self.graph = graph
        self.mask = graph.ndata['train_mask'] if is_train else graph.ndata['test_mask']
        self.label = graph.ndata['label']
        self.node = graph.nodes()
        self.feat = graph.ndata['feat'].float()

    def __len__(self):
        return self.graph.num_nodes()

    def __getitem__(self, idx):
        return {
            'node': self.node[idx],
            'y': self.label[idx],
            'mask': self.mask[idx],
            'x': self.feat[idx]
        }

def Ahat_mat(graph, config):
    A = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for from_, to_ in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        A[from_, to_] += 1
    A = A + np.identity(graph.num_nodes())
    D = np.sum(A, axis=1)
    D = np.diag(np.power(D, -0.5))
    Ahat = np.dot(D, A).dot(D)
    return torch.tensor(Ahat).float().to(config.device), A
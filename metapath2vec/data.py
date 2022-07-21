import numpy as np
import torch, gc
from torch.utils.data import Dataset

class mp2v_Dataset(Dataset):
    def __init__(self, n_node):
        super(mp2v_Dataset, self).__init__()
        self.n_node = n_node
        self.node_idx = np.random.permutation(np.arange(self.n_node))

    def __len__(self):
        return self.n_node


    def __getitem__(self, idx):

        return {'node':self.node_idx[idx]}
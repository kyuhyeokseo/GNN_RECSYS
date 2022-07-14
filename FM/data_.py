import torch
from torch.utils.data import Dataset
import numpy as np

class FMDataset(Dataset):
    # make Dataset ( user, item )

    def __init__(self, X, Y):
        self.data = torch.tensor(X, dtype = torch.float)
        self.target = torch.tensor(Y, dtype = torch.float)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        f_vector = self.data[idx,:]
        value = self.target[idx]

        return {'feature': f_vector, 'value':value}
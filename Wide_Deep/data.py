from torch.utils.data import Dataset
import numpy as np

class Dataset_(Dataset):
    def __init__(self, df, cols):
        super(Dataset_).__init__()
        self.df = df[cols].values
        self.label = df['label'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        x = self.df[i]
        y = self.label[i]

        return {'x': x, 'y': y}

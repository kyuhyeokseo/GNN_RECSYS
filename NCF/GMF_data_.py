from torch.utils.data import Dataset
import numpy as np

class GMF_Dataset(Dataset):
    def __init__(self, data, neg_data_per_pos_data):
        super(GMF_Dataset).__init__()
        self.M = data.shape[0]
        self.N = data.shape[1]
        self.data = data

        idx_mat = np.arange(self.M * self.N).reshape(self.M, self.N)

        pos_n = np.sum(data, dtype=np.int32) # 44140

        neg_idx = idx_mat[data == 0] # (1541986,)
        pos_idx = idx_mat[data == 1] # (44140,)

        neg_sampled_idx = np.random.choice(neg_idx, pos_n * neg_data_per_pos_data, replace=False)

        # 1개의 pos_data 당 4개의 neg_data가 추가되어 총 44140 x 5 = 220700의 길이를 갖는 total_rate가 sort 되어있음
        self.total_rate = np.sort(np.union1d(pos_idx, neg_sampled_idx))

    def __len__(self):
        return len(self.total_rate)

    def __getitem__(self, i):
        idx = self.total_rate[i]
        u = int(idx // self.N)
        i = int(idx % self.M)
        r = self.data[u, i]

        return (u, i, r)


class GMF_Dataset_test(Dataset):
    def __init__(self, data):
        super(GMF_Dataset_test).__init__()
        self.M = data.shape[0]
        self.N = data.shape[1]
        self.data = data

        idx_mat = np.arange(self.M * self.N).reshape(self.M, self.N)

        neg_idx = idx_mat[data == 0] # (1541986,)
        pos_idx = idx_mat[data == 1] # (44140,)

        self.total_rate = np.sort(np.union1d(pos_idx, neg_idx))

    def __len__(self):
        return len(self.total_rate)

    def __getitem__(self, i):
        idx = self.total_rate[i]
        u = int(idx // self.N)
        i = int(idx % self.M)
        r = self.data[u, i]

        return (u, i, r)
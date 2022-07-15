from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
import numpy as np

class BPRDataset(Dataset):
    # make Dataset ( data, neg_item )

    def __init__(self, data, pos_user, pos_item, neg_item, config):
        self.data = data
        self.cx = coo_matrix(data)

        self.pos_user = pos_user
        self.pos_item = pos_item
        self.neg_item = neg_item

        self.data_list = self.make_neg(self.cx, self.neg_item)
        self.config = config

    def make_neg(self, cx, neg_item):
        list = []
        for idx in range(len(self.pos_user)):
            for idx2 in range(len(neg_item[cx.row[idx]])):
                new_item = [ cx.row[idx], cx.col[idx], neg_item[cx.row[idx]][idx2] ]
                list.append(new_item)
        return list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        user = self.data_list[idx][0]
        item = self.data_list[idx][1]
        neg_item = self.data_list[idx][2]

        return {'user':user, 'item':item, 'neg_item':neg_item}





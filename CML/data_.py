from torch.utils.data import Dataset
import numpy as np

class CMLDataset(Dataset):
    # make Dataset ( user, item, neg_item_sample_for_user, item_details(features) )

    def __init__(self, all_data, item_data, neg_size, neg_item):
        self.all_data = all_data
        self.item_data = item_data
        self.pos_user = all_data.nonzero()[0]
        self.pos_item = all_data.nonzero()[1]
        self.n_user , self.n_item = all_data.shape
        self.each_neg_sample_size = neg_size
        self.neg_item = neg_item

    def __len__(self):
        return len(self.pos_user)

    def __getitem__(self, idx):
        user = self.pos_user[idx]
        item = self.pos_item[idx]
        neg_item_sample = np.random.choice(self.neg_item[user], self.each_neg_sample_size)
        item_detail = self.item_data[item,:]
        return {'user':user, 'item':item, 'neg_item':neg_item_sample, 'item_detail':item_detail}





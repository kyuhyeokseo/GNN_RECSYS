import numpy as np
import torch, gc
from torch.utils.data import Dataset

class KgDataset(Dataset):
    def __init__(self, head, tail, label, neg_sample_k, head_neg, tail_neg):
        super(KgDataset, self).__init__()
        self.head = head
        self.tail = tail
        self.label = label
        self.neg_sample_k = neg_sample_k
        self.head_neg = head_neg
        self.tail_neg = tail_neg

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        head, tail, label = self.head[idx], self.tail[idx], self.label[idx]
        tail_prime = np.random.choice(self.head_neg[head], self.neg_sample_k)
        head_prime = np.random.choice(self.tail_neg[tail], self.neg_sample_k)

        return {'head':head, 'label':label, 'tail':tail, 'tail_p':tail_prime, 'head_p':head_prime}



def prepare_neg_entity(head, tail, label):
    adj_mat = np.zeros((np.max(head)+1, np.max(head)+1))
    for h,t,l in zip(head, tail, label):
        adj_mat[h,t] = l

    idx = torch.arange(np.max(head)+1)

    # head_neg [h,t] : head h 에 대해 실제 relation을 가지지 않는 tail t의 모임
    # tail_neg [h,t] : tail t 에 대해 실제 relation을 가지지 않는 head h의 모임
    head_neg = {h:idx[~adj_mat[h].astype(bool)] for h in np.unique(head)}
    tail_neg = {t:idx[~adj_mat[:, t].astype(bool)] for t in np.unique(tail)}

    # adj_mat 가 차지하는 memory가 엄청나므로, 이를 지우기 위해서 해당 작업 실시
    del adj_mat; gc.collect()


    return head_neg, tail_neg

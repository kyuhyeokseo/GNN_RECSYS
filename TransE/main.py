import numpy as np
import torch
from torch import IntTensor
from torch.utils.data import DataLoader
import torch.optim as optim
from data_ import KgDataset, prepare_neg_entity
from model_ import TransE

data_PATH = 'data/'

head = torch.load(data_PATH + 'KG_head.pt').numpy()
tail = torch.load(data_PATH + 'KG_tail.pt').numpy()
label = torch.load(data_PATH + 'KG_label.pt').numpy()

neg_size_each = 10  # h'와 t' 각각 count, 결국 head_neg, tail_neg가 각각 size가 10

head_neg, tail_neg = prepare_neg_entity(head, tail, label)
dataset = KgDataset(head, tail, label, neg_size_each, head_neg, tail_neg)

batch_size = 256

# DataLoader에 들어갈 dataset 부분은 __len__, __getitem__ 함수를 가지고 있어야 하므로
# data_.py 를 보면 KgDataset class 의 함수에 해당 함수들 정의하였음
# 1개의 batch 당 config.batch_size = 1024 개의 데이터가 들어감
# shuffle = True 이므로 dataset에서 가지던 순서가 섞이되, 같은 쌍끼리는 같은 idx 유지
dataloader = DataLoader(dataset, batch_size, drop_last=False, shuffle=True)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
gamma = 2
embed_dim = 20

# head, tail은 0이 아닌, 1부터 시작지점이므로 +1 작업
model = TransE(num_entity = max(head) + 1, num_label=max(label) + 1,
               embed_dim = embed_dim, gamma = gamma, device = device)

model = model.to(device)

learning_rate = 0.01
weight_decay = 0.01
epochs = 50

optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

for epoch in range( epochs ):
    # Train ON
    model.train()

    for batch_data in dataloader:
        # gradient initializing
        optimizer.zero_grad()

        batch_data = {k: v.to(device) for k, v in batch_data.items()}

        loss = model(batch_data)
        loss.backward()
        optimizer.step()


for batch_data in dataloader :

    batch_head, batch_label, batch_tail = batch_data['head'], batch_data['label'], batch_data['tail']
    batch_head_p= batch_data['head_p']

    size = len(batch_head_p)
    hit10 = 0
    rank_list = []

    for idx in range(size):
        all, rank = 0, 0
        h_tensor, l_tensor, t_tensor = batch_head[idx], batch_label[idx], batch_tail[idx]
        h, t = IntTensor.item(batch_head[idx]), IntTensor.item(batch_tail[idx])

        for h_prime_tensor in head_neg[h]:
            all += 1
            if model.distance_l2(h_prime_tensor, l_tensor, t_tensor) < model.distance_l2(h_tensor, l_tensor, t_tensor) :
                rank += 1

        rank_list.append(rank)
        if rank <= 10 :
            hit10 += 1
        all, rank = 0, 0


        for t_prime_tensor in tail_neg[t]:
            all += 1
            if model.distance_l2(h_tensor, l_tensor, t_prime_tensor) < model.distance_l2(h_tensor, l_tensor, t_tensor) :
                rank += 1

        rank_list.append(rank)
        if rank <= 10 :
            hit10 += 1



    # Raw DATA : for one batch_data
    print(np.mean(rank_list)) # 326.64 mean rank in my test
    print(hit10/size) # 71.9 % HITS@10 in my test

    # For all batch_data sets, delete "break" below
    break









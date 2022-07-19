import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from data import Dataset_
from model import Wide_Deep


def cross_data(df, cross_cols):
    new_cols = []
    for a in cross_cols:
        col_name = '-'.join(a)
        df[col_name] = df[a].astype(str).apply(lambda x: ''.join(x), axis=1)
        new_cols.append(col_name)

    return df, new_cols

def categorical_encoding(data, cols):
    encoder_record = {}
    for col in cols:
        encoder = LabelEncoder()
        encoder.fit(data[col].values)
        encoder_record[col] = encoder

    for col, enc in encoder_record.items():
        data[col] = enc.transform(data[col])

    return encoder_record, data

def continuous_encoding(data, cols):
    encoder = StandardScaler()
    encoder.fit(data[cols].values)
    data[cont_cols] = encoder.transform(data[cols].values)

    return encoder, data


def get_wide_data(df, new_cols, cat_cols, cont_cols, target):
    oh_df = pd.get_dummies(df[new_cols + cat_cols])
    cont_encoder, df = continuous_encoding(df, cont_cols)

    df = pd.concat([df[cont_cols + [target] + ['is_train']], oh_df], axis=1)
    model_var = [col for col in df if col not in [target]]

    return df, model_var


def get_deep_data(df, new_cols, cat_cols, cont_cols, target):
    cont_encoder, df = continuous_encoding(df, cont_cols)
    cat_encoders, df = categorical_encoding(df, cat_cols + new_cols)

    df = df[cat_cols + new_cols + cont_cols + [target] + ['is_train']]

    model_var = [col for col in df if col not in ['label', 'is_train']]

    return df, model_var, cat_encoders

class Config():
    columns = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
    # category, continuous, cross column
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender',
                'native_country', ]
    cont_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', ]
    cross_cols = ['education', 'occupation'], ['native_country', 'occupation']
    train_col = ['is_train']
    target = 'label'  # target data from column 'income_bracket'
    batch_size = 256
    embed_dim = 64
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    weight_decay = 0.001
    epochs = 20

config = Config()

COLUMNS = config.columns

df_train = pd.read_csv("dataset/adult.data", names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv("dataset/adult.test", names=COLUMNS, skipinitialspace=True, skiprows=1)

# income_label추가
df_train['label'] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test['label'] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_train = df_train.drop(['income_bracket'], axis = 1)
df_test = df_test.drop(['income_bracket'], axis = 1)

cross_cols = config.cross_cols
cat_cols = config.cat_cols
cont_cols = config.cont_cols

df_train['is_train'] = 1
df_test['is_train'] = 0

full_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# make cross data
full_df, new_cols = cross_data(full_df, cross_cols)

df_wide, wide_cols = get_wide_data(full_df, new_cols, cat_cols, cont_cols, config.target)
df_deep, deep_cols, cat_encoders = get_deep_data(full_df, new_cols, cat_cols, cont_cols, config.target)

train_mask_wide = df_wide['is_train'] == 1
train_wide = df_wide[train_mask_wide]
test_wide = df_wide[~train_mask_wide]
train_mask_deep = df_deep['is_train'] == 1
train_deep = df_deep[train_mask_deep]
test_deep = df_deep[~train_mask_deep]


dataset_train_wide = Dataset_(train_wide, wide_cols)
dataset_test_wide = Dataset_(test_wide, wide_cols)
dataset_train_deep = Dataset_(train_deep, deep_cols)
dataset_test_deep = Dataset_(test_deep, deep_cols)

dataloader_train_wide = DataLoader(dataset_train_wide, batch_size=config.batch_size, drop_last=False, shuffle=True)
dataloader_test_wide = DataLoader(dataset_test_wide, batch_size=config.batch_size, drop_last=False, shuffle=False)
dataloader_train_deep = DataLoader(dataset_train_deep, batch_size=config.batch_size, drop_last=False, shuffle=True)
dataloader_test_deep = DataLoader(dataset_test_deep, batch_size=config.batch_size, drop_last=False, shuffle=False)

# cat_cols : 8, new_cols : 2 -> need Embedding , cont_cols : 6, 'label' : 1

model = Wide_Deep(cat_cols, new_cols, cont_cols, cat_encoders, config, wide_cols)
model = model.to(config.device)

optimizer = optim.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
loss_fn = nn.BCEWithLogitsLoss()


for epoch in range( config.epochs ):
    # Train ON
    model.train()

    loss_train_list = []
    for batch_wide, batch_deep in zip(dataloader_train_wide, dataloader_train_deep):
        # gradient initializing
        optimizer.zero_grad()

        batch_data_wide = {k: v.to(config.device) for k, v in batch_wide.items()}
        batch_data_deep = {k: v.to(config.device) for k, v in batch_deep.items()}

        wide_x, wide_y = batch_data_wide['x'].to(dtype = torch.float), batch_data_wide['y'].to(dtype = torch.float)
        deep_x, deep_y = batch_data_deep['x'].to(dtype=torch.float), batch_data_deep['y'].to(dtype=torch.float)

        pred = model(wide_x, deep_x)
        loss = loss_fn(pred, wide_y.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        loss_train_list.append(loss.item())

    loss_test_list = []
    for batch_wide, batch_deep in zip(dataloader_test_wide, dataloader_test_deep):

        batch_data_wide = {k: v.to(config.device) for k, v in batch_wide.items()}
        batch_data_deep = {k: v.to(config.device) for k, v in batch_deep.items()}

        wide_x, wide_y = batch_data_wide['x'].to(dtype=torch.float), batch_data_wide['y'].to(dtype=torch.float)
        deep_x, deep_y = batch_data_deep['x'].to(dtype=torch.float), batch_data_deep['y'].to(dtype=torch.float)

        pred = model(wide_x, deep_x)
        loss = loss_fn(pred, wide_y.unsqueeze(-1))
        if epoch == config.epochs -1 :
            with torch.no_grad():
                fpr, tpr, _ = roc_curve(wide_y.squeeze(-1).numpy(), pred.numpy())

        loss_test_list.append(loss.item())

    print(f'EPOCH {epoch + 1} : TRAIN BCE_LogLoss {np.mean(loss_train_list) : .4f}, TEST BCE_LogLoss {np.mean(loss_test_list) : .4f}')


# Plotting
plt.plot(fpr, tpr, marker='.')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
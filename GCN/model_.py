import torch
import torch.nn as nn
import torch.nn.functional

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()

        self.input = config.input_dim
        self.output = config.output_dim
        self.dropout = nn.Dropout(0.1)
        self.W1 = nn.Linear(self.input, config.hidden_dim)
        self.W2 = nn.Linear(config.hidden_dim, self.output)

    def forward(self, batch_data, A_hat):

        label, data, mask = batch_data['y'], batch_data['x'], batch_data['mask']

        output1 = torch.spmm(A_hat, data)
        output1 = self.dropout(output1)
        output1 = self.W1(output1)

        output2 = torch.nn.functional.relu(output1)

        output2 = torch.spmm(A_hat, output2)
        output2 = self.dropout(output2)
        output2 = self.W2(output2)

        return output2[mask], label[mask]

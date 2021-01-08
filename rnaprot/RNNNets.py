import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class RNNDataset(Dataset):
    def __init__(self, in_data, in_labels):
        self.data = in_data
        self.labels = in_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, n_class, device):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layer, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, n_class)
        self.device = device

    def zero_state(self, batch_size):
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
        c0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
        return h0, c0

    def forward(self, embed):
        embed = embed.cuda()
        batch_size = embed.batch_sizes[0].item()
        h0, c0 = self.zero_state(batch_size)
        out, (hidden, cell) = self.lstm(embed, (h0, c0))
        x = F.log_softmax(self.fc(hidden.float()), dim=-1)
        return x


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, n_class, device,
                 dr=0.5):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.gru = nn.GRU(input_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_class)
        self.device = device
        self.dr = dr

    def zero_state(self, batch_size):
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_dim, dtype=torch.float).requires_grad_().to(self.device)
        return h0

    def forward(self, embed):
        embed = embed.cuda()
        batch_size = embed.batch_sizes[0].item()
        h0 = self.zero_state(batch_size)
        out, hidden = self.gru(embed, h0.detach())
        x = F.dropout(hidden, p=self.dr, training=self.training)
        x = F.log_softmax(self.fc(x), dim=-1)
        return x

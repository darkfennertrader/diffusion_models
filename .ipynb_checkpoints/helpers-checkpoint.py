import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

def sample_batch(batch_size, device = "cpu"):
    data, _ = make_swiss_roll(batch_size, random_state=42)
    # operations to make it equal to the paper
    data = data[:, [2,0]] / 10
    # flip the image vertically by inverting the rows in reverse order
    data = data*np.array([1, -1])
    return torch.from_numpy(data).to(device)


class MLP(nn.Module):
    def __init__(self, N=40, data_dim=2, hidden_dim=64):
        super(MLP, self).__init__()
        self.network_head = nn.Sequential(nn.Linear(data_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),)
        self.network_tail = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, data_dim * 2),) for t in range(N)])

    def forward(self, x, t):
        h = self.network_head(x) # [batch_size, hidden_dim]
        #print(h.shape)
        tmp = self.network_tail[t](h) # [batch_size, data_dim * 2]
        #print(tmp.shape)
        mu, h = torch.chunk(tmp, 2, dim=1)
        var = torch.exp(h)
        std = torch.sqrt(var)
        
        return mu, std 
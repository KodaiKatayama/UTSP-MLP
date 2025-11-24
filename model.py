import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, alpha):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_nodes)
        self.alpha = alpha

    def forward(self, x):
        h = self.embed(x)
        h = F.relu(h)
        raw_logits = self.output(h)
        logits = self.alpha * torch.tanh(raw_logits)
        return logits
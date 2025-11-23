import torch
import torch.nn as nn

class TSPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super().__init__()
        # 座標(2次元)を隠れ層のサイズに変換
        self.embed = nn.Linear(input_dim, hidden_dim)
        # 最終的に N x N の行列にするための層
        self.output = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x):
        # x: (バッチ, 都市数, 2)
        h = self.embed(x)             # (バッチ, 都市数, hidden)
        logits = self.output(h)       # (バッチ, 都市数, 都市数)
        return logits
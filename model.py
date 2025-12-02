import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
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
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, alpha, **kwargs):
        super().__init__()
        self.alpha = alpha
        # TODO: GCNのレイヤー定義をここに書く
        # self.conv1 = ...
        pass

    def forward(self, x, adj):
        """
        GNN系は adj (隣接行列) を使ってメッセージパッシングを行う
        """
        # TODO: 実装
        # h = self.conv1(x, adj)
        # ...
        
        # ダミー出力: MLPと同様の形状を返す必要がある (Batch, N, N)
        batch_size, n_nodes, _ = x.shape
        raw_logits = torch.zeros(batch_size, n_nodes, n_nodes, device=x.device)
        
        logits = self.alpha * torch.tanh(raw_logits)
        return logits

class SAGs(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, alpha, **kwargs):
        super().__init__()
        self.alpha = alpha
        # TODO: Scattering Attention Graph Neural Network の定義
        pass

    def forward(self, x, adj):
        # TODO: 実装
        batch_size, n_nodes, _ = x.shape
        raw_logits = torch.zeros(batch_size, n_nodes, n_nodes, device=x.device)
        
        logits = self.alpha * torch.tanh(raw_logits)
        return logits

def get_model(model_name, input_dim, hidden_dim, num_nodes, alpha, **kwargs):
    """
    モデル名に応じてインスタンスを返すファクトリー関数
    """
    model_name = model_name.lower()
    
    if model_name == 'mlp':
        return MLP(input_dim, hidden_dim, num_nodes, alpha, **kwargs)
    elif model_name == 'gcn':
        return GCN(input_dim, hidden_dim, num_nodes, alpha, **kwargs)
    elif model_name == 'sags':
        return SAGs(input_dim, hidden_dim, num_nodes, alpha, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSinkhorn(nn.Module):
    def __init__(self, iterations=60, temperature=1.0):

        super().__init__()
        self.iterations = iterations
        self.temperature = temperature

    def forward(self, logits):
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        
        y = (logits + gumbel_noise) / self.temperature
        
        P = F.softmax(y, dim=-1)
        
        for _ in range(self.iterations):
            P = P / (P.sum(dim=1, keepdim=True) + 1e-20)
            P = P / (P.sum(dim=2, keepdim=True) + 1e-20)
            
        return P

class SimpleTSPModel(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.encoder = nn.Linear(1, 1) 
        self.sinkhorn = GumbelSinkhorn(iterations=60, temperature=1.0)
        self.alpha = nn.Parameter(torch.tensor(10.0))

    def forward(self, distances):
        x = distances.unsqueeze(-1)
        logits = self.encoder(x).squeeze(-1)
        logits = self.alpha * torch.tanh(logits)
        soft_perm = self.sinkhorn(logits)
        
        return soft_perm
    
if __name__ == "__main__":
    dummy_dist = torch.rand(2, 5, 5)
    model = SimpleTSPModel(num_nodes=5)
    output_matrix = model(dummy_dist)
    
    print("入力の形:", dummy_dist.shape)
    print("出力の形:", output_matrix.shape)
    print("\n--- 1つ目のデータの、各行の合計を確認 ---")
    row_sums = output_matrix[0].sum(dim=1)
    print(row_sums)
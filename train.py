import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import TSPDataset
from model import TSPModel
from utils import get_cyclic_matrix, sample_gumbel, sinkhorn, Hungarian

# --- 設定エリア ---
NUM_NODES = 20        # 都市の数
NUM_SAMPLES = 1000    # データ数
BATCH_SIZE = 32       # 一度に計算する数 (不明)
EPOCHS = 100          # 学習回数
LR = 1e-3             # 学習率
ALPHA = 10.0          # スケーリング定数 (α)
TAU = 2.0             # 温度パラメータ (tau) 
GAMMA = 0.1           # ノイズの大きさ (gamma)
SINKHORN_ITERS = 60   # Sinkhornの繰り返し回数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = TSPDataset(NUM_SAMPLES, NUM_NODES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TSPModel(input_dim=2, hidden_dim=128, num_nodes=NUM_NODES, alpha=ALPHA).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
V = get_cyclic_matrix(NUM_NODES).to(DEVICE)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    total_real = 0
    
    for batch in dataloader:
        points = batch['points'].to(DEVICE)
        distances = batch['distance'].to(DEVICE)
        
        optimizer.zero_grad()

        logits = model(points)
        
        noise = sample_gumbel(logits.shape).to(DEVICE)
        noisy_logits = (logits + GAMMA * noise) / TAU
        T = sinkhorn(noisy_logits, n_iters=SINKHORN_ITERS)
        
        V_batch = V.unsqueeze(0) 
        soft_adj = torch.matmul(torch.matmul(T, V_batch), T.transpose(1, 2))
        loss = torch.sum(distances * soft_adj) / BATCH_SIZE
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # --- ハード割当をバッチ単位で評価して累積 ---
        with torch.no_grad():
            P = Hungarian(-noisy_logits)
            V_batch = V.unsqueeze(0)
            hard_adj = torch.matmul(torch.matmul(P, V_batch), P.transpose(1, 2))
            current_dist_batch = torch.sum(distances * hard_adj, dim=(1, 2))
            total_real += current_dist_batch.sum().item()
    # エポック中に累積したハード距離の平均（サンプルあたり）
    dataset_size = len(dataset)
    mean_real_dist = total_real / dataset_size

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}, Real Distance = {mean_real_dist:.4f}")
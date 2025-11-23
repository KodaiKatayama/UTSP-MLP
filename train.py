import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import TSPDataset
from model import TSPModel
from utils import get_cyclic_matrix, sample_gumbel, sinkhorn

# --- 設定エリア ---
NUM_NODES = 20        # 都市の数
NUM_SAMPLES = 1000    # データ数
BATCH_SIZE = 32       # 一度に計算する数
EPOCHS = 100          # 学習回数
LR = 1e-3             # 学習率
TAU = 2.0            # 温度パラメータ (tau)
GAMMA = 0.1           # ノイズの大きさ (gamma)
SINKHORN_ITERS = 60   # Sinkhornの繰り返し回数
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = TSPDataset(NUM_SAMPLES, NUM_NODES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TSPModel(input_dim=2, hidden_dim=128, num_nodes=NUM_NODES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 正解のリング行列 V を作る (GPUへ送る)
V = get_cyclic_matrix(NUM_NODES).to(DEVICE)

# 2. 学習ループ
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch in dataloader:
        # データをGPUへ
        points = batch['points'].to(DEVICE)     # 座標
        distances = batch['distance'].to(DEVICE)# 距離行列 D
        
        optimizer.zero_grad()
        
        # (1) モデルでスコア(logits)を出す
        logits = model(points)
        
        # (2) ガンベルノイズを加えて Sinkhorn で確率行列 T にする
        #     式(8): (F + gamma * noise) / TAU
        noise = sample_gumbel(logits.shape).to(DEVICE)
        noisy_logits = (logits + GAMMA * noise) / TAU
        T = sinkhorn(noisy_logits, n_iters=SINKHORN_ITERS)
        
        # (3) 損失関数の計算: Loss = <D, T V T^T>
        #     まず "T V T^T" を計算 (ソフトな隣接行列を作る)
        #     T(バッチ,N,N) @ V(N,N) @ T_transpose(バッチ,N,N)
        #     ※ Vはバッチがないので unsqueeze(0) で (1,N,N) にして合わせる
        V_batch = V.unsqueeze(0) 
        soft_adj = torch.matmul(torch.matmul(T, V_batch), T.transpose(1, 2))
        
        #     距離行列 D との内積をとる (要素ごとの掛け算の合計)
        #     論文式(5) [cite: 633]
        loss = torch.sum(distances * soft_adj) / BATCH_SIZE
        
        # (4) 更新
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
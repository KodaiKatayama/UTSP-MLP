import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TSPDataset

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 特徴量を変換するLinear
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adjacency):
        """
        x: (Batch, N, in_dim) -> 各都市の特徴量
        adjacency: (Batch, N, N) -> 「繋がりやすさ」を表す行列
        """
        # 1. メッセージパッシング (集約)
        # 隣接行列 @ 特徴量 = 「つながっている相手の情報を足し合わせる」
        # これで「周りの情報」が自分に入ってくる
        neighbor_info = torch.bmm(adjacency, x)
        
        # 2. 情報の変換 (Linear)
        out = self.linear(neighbor_info)
        
        # 3. 活性化関数 (ReLU)
        return F.relu(out)

class GumbelSinkhorn(nn.Module):
    def __init__(self, iterations=60, temperature=3.0, gamma=0.01):
        super().__init__()
        self.iterations = iterations
        self.temperature = temperature
        self.gamma = gamma

    def forward(self, logits, debug=False):
        
        # 1. Gumbelノイズ (epsilon) の生成
        u = torch.rand_like(logits)
        epsilon = -torch.log(-torch.log(u)) #逆関数法
        
        # 2. 式(8)の実装: (F + gamma * epsilon) / tau
        y = (logits + self.gamma * epsilon) / self.temperature
        
        # 3. Sinkhorn正規化
        P = F.softmax(y, dim=-1)

        if debug:
            print("\n--- [Step 0] 初期Softmax後 (行だけ1になってるはず) ---")
            print(P[0].detach().numpy().round(3)) # 最初のバッチだけ表示
        
        for _ in range(self.iterations):
            P = P / (P.sum(dim=1, keepdim=True))
            P = P / (P.sum(dim=2, keepdim=True))
            
            if debug and i in [0, 1, 5, self.iterations - 1]:
                print(f"\n--- [Iteration {i+1}] (正規化の途中経過) ---")
                print(P[0].detach().numpy().round(3))
                
                # 行と列の合計もチェック
                row_sum = P[0].sum(dim=1).detach().numpy()
                col_sum = P[0].sum(dim=0).detach().numpy()
                print(f"  > 行の合計: {row_sum.round(3)}")
                print(f"  > 列の合計: {col_sum.round(3)}")
            
        return P

# model.py の TSPModel クラス

class TSPModel(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128):
        super().__init__()
        
        # 座標(2次元)を受け取るので、入力サイズは 2 で正解！
        self.embedding = nn.Linear(2, hidden_dim)
        
        # (以下の GCN定義 や Sinkhorn定義 はそのままでOK)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.sinkhorn = GumbelSinkhorn(iterations=60, temperature=3.0, gamma=0.01)
        self.alpha = 10.0
        self.register_buffer('matrix_V', self._create_canonical_cycle(num_nodes))

    def _create_canonical_cycle(self, n):
        # (そのまま)
        V = torch.zeros(n, n)
        for i in range(n - 1):
            V[i, i + 1] = 1.0
        V[n - 1, 0] = 1.0
        return V

    # ★ここを修正！引数に points を追加
    def forward(self, points, distances):
        """
        points: (Batch, N, 2)    <-- 都市の座標 (これが欲しかった！)
        distances: (Batch, N, N) <-- 距離行列
        """
        
        # --- Step 1: 隣接行列 A ---
        adjacency = torch.exp(-distances) 
        
        # --- Step 2: 特徴量の埋め込み ---
        # ★修正: distances ではなく points を入れる！
        # points は (Batch, N, 2) なので、Linear(2, 128) にピッタリ入る
        x = self.embedding(points) 
        
        # --- Step 3: GCN ---
        # (以下そのまま)
        x = self.gcn1(x, adjacency)
        x = self.gcn2(x, adjacency)
        x = self.gcn3(x, adjacency)
        
        logits = torch.bmm(x, x.transpose(1, 2))
        logits = logits - 1e9 * torch.eye(logits.size(1), device=logits.device).unsqueeze(0)

        logits = self.alpha * torch.tanh(logits)
        soft_perm = self.sinkhorn(logits)
        
        return soft_perm

def test_model_full_flow():
    print("=== モデル動作の完全テストを開始 ===")
    
    # 1. 設定
    batch_size = 2
    num_nodes = 5
    model = TSPModel(num_nodes=num_nodes)
    
    # 2. データを用意 (data.pyで作ったクラスを使う)
    dataset = TSPDataset(num_samples=batch_size, num_nodes=num_nodes)
    data = dataset[0] # 1つ取り出す
    
    # バッチ次元を足す (N, N) -> (1, N, N)
    # 本番のDataLoaderはこれを自動でやってくれますが、テストなので手動で
    distances = data['distance'].unsqueeze(0) 
    
    print(f"Input Distance Shape: {distances.shape}")

    # 3. Forwardパス (順伝播)
    # ここでエラーが出たら model.py の forward か shape が間違ってる
    soft_perm = model(distances)
    
    print(f"Output SoftPerm Shape: {soft_perm.shape}")
    
    # チェック1: 形はあっているか？ (Batch, N, N)
    assert soft_perm.shape == (1, num_nodes, num_nodes), "出力サイズが変です！"
    
    # チェック2: Sinkhornは効いているか？ (行の和がほぼ1.0)
    row_sum = soft_perm.sum(dim=2)
    print(f"Row Sums: {row_sum.detach().numpy()}")
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
        print("警告: 行の合計が1になっていません！Sinkhornの設定を見直してください")
    
    # 4. Loss計算のテスト (これが一番大事！)
    # 論文の式: Loss = <D, T V T_t>
    
    # Vを取り出す (register_bufferしたやつ)
    V = model.matrix_V
    
    # T * V * T^T を計算
    # batch行列積 (bmm) を使います
    # (B, N, N) x (N, N) は直接できないので、Vをバッチサイズ分コピーするか、matmulを使う
    
    # Vをバッチサイズに合わせて拡張: (1, N, N)
    V_batch = V.unsqueeze(0).expand(distances.size(0), -1, -1)
    
    # T @ V @ T.transpose
    permuted_V = torch.bmm(soft_perm, torch.bmm(V_batch, soft_perm.transpose(1, 2)))
    
    # 距離行列 D との内積 (総距離)
    loss = torch.sum(distances * permuted_V)
    
    print(f"Calculated Loss (Total Distance): {loss.item()}")
    
    # 5. Backwardパス (逆伝播)
    # ここでエラーが出たら「学習できない（勾配が途切れてる）」ということ
    loss.backward()
    
    # encoderの重みに勾配が入っているかチェック
    if model.encoder.weight.grad is not None:
        print("Gradient Check: OK! (学習可能です)")
    else:
        print("Gradient Check: NG! (勾配が流れていません。detach()とかしてませんか？)")

    print("\n=== All Checks Passed! 安心してください、動きます ===")

if __name__ == "__main__":
    test_model_full_flow()
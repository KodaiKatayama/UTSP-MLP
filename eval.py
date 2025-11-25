import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.backends.backend_pdf as pdf_backend # PDF出力のためのライブラリ

from data import TSPDataset
from model import TSPModel
from utils import get_cyclic_matrix, Hungarian

# --- 設定エリア ---
NUM_NODES = 20
TEST_SAMPLES = 1000 
BATCH_SIZE = 32
TAU = 2.0
ALPHA = 10.0
SAVE_PATH = 'best_model.pth' 
HISTORY_FILE = 'training_history.npz'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 論文のベンチマーク値 (TSP-20) ---
OPTIMAL_LEN = 3.83 
GREEDY_NN_LEN = 4.51
# PAPER_MODEL_LEN = 4.06 (あなたのモデル結果と比較するため使用しません)
# -------------------------------------

def get_test_distances(model, V_matrix):
    """ベストモデルをロードし、テストデータ1000問の距離をリストで取得する"""
    dataset = TSPDataset(TEST_SAMPLES, NUM_NODES) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_distances = []
    
    with torch.no_grad():
        for batch in dataloader:
            # ... (中略: データロードのロジックは変更なし)
            points = batch['points'].to(DEVICE)
            distances = batch['distance'].to(DEVICE)
            logits = model(points)
            
            # P = Hungarian(-F/τ)
            P = Hungarian(-(logits / TAU)) 
            
            V_batch = V_matrix.unsqueeze(0)
            hard_adj = torch.matmul(torch.matmul(P, V_batch), P.transpose(1, 2))
            
            dist_per_instance = torch.sum(distances * hard_adj, dim=(1, 2))
            all_distances.extend(dist_per_instance.cpu().numpy().tolist())
            
    return np.array(all_distances)

def plot_results(model):
    # 履歴をロード (図1用)
    try:
        history = np.load(HISTORY_FILE)
    except FileNotFoundError:
        print("Error: Training history not found. Run train.py first.")
        return

    # ----------------------------------------------
    # 1. 図1: 学習履歴のプロット (PNG出力)
    # ----------------------------------------------
    
    # 論文の図1の縦横比 (概ね 3:2) を参考に設定
    plt.figure(figsize=(6, 4)) 
    plt.plot(history['loss'], label='Training Loss', color='blue')
    plt.plot(history['dist'], label='Real Distance (Evaluation)', color='red')
    plt.title(f'Figure 1: Training History (TSP-{NUM_NODES})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figure1_TrainingHistory.png', bbox_inches='tight') # PNGとして保存
    plt.close() # グラフウィンドウを閉じる

    # ----------------------------------------------
    # 2. 図2の再現 (ヒストグラム) と PDF出力
    # ----------------------------------------------
    
    V = get_cyclic_matrix(NUM_NODES).to(DEVICE)
    test_distances = get_test_distances(model, V)
    
    # PDFファイルを生成
    pdf_pages = pdf_backend.PdfPages('Figure2_TourLengthDistribution.pdf')

    # 論文の図2（左側）の縦横比に設定 (概ね 1:1.2 ~ 1:1.5)
    fig = plt.figure(figsize=(5, 6.5)) 
    
    # ヒストグラムの描画
    plt.hist(test_distances, bins=20, density=True, alpha=0.6, 
             label=f'Current Model ($\mu$={np.mean(test_distances):.4f})', 
             color='purple')
    
    # ベンチマーク値を縦線で表示
    plt.axvline(OPTIMAL_LEN, color='green', linestyle='--', label=f'Optimal ($\mu$={OPTIMAL_LEN})')
    plt.axvline(GREEDY_NN_LEN, color='red', linestyle='--', label=f'Greedy NN ($\mu$={GREEDY_NN_LEN})')
    
    plt.title(f'Figure 2: Tour Length Distribution (TSP-{NUM_NODES})')
    plt.xlabel('Tour Length')
    plt.ylabel('Density (Counts)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # PDFとして図を保存
    pdf_pages.savefig(fig, bbox_inches='tight')
    pdf_pages.close() # PDFファイルを閉じる

    print("\n--- 描画結果 ---")
    print("1. 学習履歴: Figure1_TrainingHistory.png に保存されました。")
    print("2. 距離分布: Figure2_TourLengthDistribution.pdf に保存されました。")
    print(f"平均ツアー長: {np.mean(test_distances):.4f} (目標: 4.06)")


if __name__ == "__main__":
    # ... (実行ロジックは変更なし)
    model = TSPModel(input_dim=2, hidden_dim=128, num_nodes=NUM_NODES, alpha=ALPHA).to(DEVICE)
    V = get_cyclic_matrix(NUM_NODES).to(DEVICE) # Vを初期化

    try:
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        plot_results(model)
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"Model file not found at {SAVE_PATH}. Run train.py first to save the model.")
        print("-------------")
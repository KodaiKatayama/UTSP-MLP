import torch
import sys
import os
import pytest

# 親ディレクトリのモジュール(data, utils)をインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import TSPDataset
from utils import get_adjacency_matrix

def test_pipeline_generation_to_adjacency():
    # --- 1. 設定 ---
    n_nodes = 4      # 見やすいように少なく設定
    scale_s = 1.0    # 論文のパラメータ s
    
    # --- 2. データ生成 (data.py) ---
    # seed=42で固定して、毎回同じ結果が出るようにする
    dataset = TSPDataset(num_samples=1, num_nodes=n_nodes, seed=42)
    sample = dataset[0]
    
    points = sample['points']    # 座標 x
    dist_D = sample['distance']  # 距離行列 D

    # --- 3. 変換処理 (utils.py) ---
    # 式(6): A = exp(-D / s)
    adj_A = get_adjacency_matrix(dist_D, scale=scale_s)

    # --- 4. 目視確認用の出力 (-s オプションで見える) ---
    print(f"\n\n=== パイプライン確認 (Nodes={n_nodes}, s={scale_s}) ===")
    
    print("\n[Step 1] 都市の座標:")
    for i, (x, y) in enumerate(points):
        print(f"  City {i}: ({x:.4f}, {y:.4f})")

    print("\n[Step 2] 距離行列 D (対角成分は0):")
    # 見やすくフォーマット
    for row in dist_D:
        print("  " + "  ".join(f"{val:6.4f}" for val in row))

    print("\n[Step 3] 隣接行列 A (対角成分は1, 遠いほど0):")
    for row in adj_A:
        print("  " + "  ".join(f"{val:6.4f}" for val in row))
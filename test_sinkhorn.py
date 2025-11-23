import torch
import sys
import os

# 親ディレクトリのファイルをimportできるようにパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model import GumbelSinkhorn

def test_visualize_sinkhorn_process():
    print("=== Sinkhornの進化を観察するテスト ===")
    
    # 見やすいように 3x3 の小さな行列で実験
    # バッチサイズ=1
    # わかりやすく手動で値を設定してみる (対角成分が大きめなケース)
    logits = torch.tensor([[[10.0, 2.0, 1.0],
                            [ 2.0, 8.0, 3.0],
                            [ 1.0, 5.0, 9.0]]])
    
    print(f"入力 Logits:\n{logits[0].numpy()}")

    # モデル準備 (debugモード用)
    # gamma=0 にしてノイズを消すと、純粋なSinkhornの動きが見れます
    sinkhorn = GumbelSinkhorn(iterations=10, temperature=1.0, gamma=0.0)

    print("\n計算開始...")
    # debug=True にして内部を覗き見る！
    output = sinkhorn(logits, debug=True)

    print("\n=== 最終結果 ===")
    print(output[0].detach().numpy().round(3))

if __name__ == "__main__":
    # 見やすくするために表示桁数を制限
    torch.set_printoptions(sci_mode=False, precision=3)
    test_visualize_sinkhorn_process()
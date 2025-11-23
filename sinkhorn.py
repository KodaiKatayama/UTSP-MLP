import torch
import torch.nn as nn

def sample_gumbel(shape, eps=1e-20):
    """
    Gumbel(0, 1) 分布からサンプリングするヘルパー関数
    Formula: -log(-log(U)) where U ~ Uniform(0, 1)
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sinkhorn(logits, tau=1.0, n_iters=60, noise_std=0.01, training=True):
    """
    論文 式(8): Gumbel-Sinkhorn Operator
    
    Args:
        logits (torch.Tensor): GNNからの出力 [batch_size, n, n]
        tau (float): 温度パラメータ (temperature)
        n_iters (int): Sinkhorn反復回数 (l)
        noise_std (float): ノイズの大きさ (gamma)
        training (bool): 学習中かどうか (推論時はノイズなし)
    
    Returns:
        torch.Tensor: 二重確率行列 (Doubly Stochastic Matrix) [batch_size, n, n]
                      行の和も列の和もほぼ1になる
    """
    # 1. Gumbelノイズの付与
    # 学習中(training=True)のみノイズを加えるのが一般的
    if training:
        gumbel_noise = sample_gumbel(logits.shape).to(logits.device)
        # 式(8)の分子: F + gamma * epsilon
        noisy_logits = logits + (noise_std * gumbel_noise)
    else:
        noisy_logits = logits

    # 2. 温度パラメータでスケーリングして指数をとる
    # Sinkhornは正の行列に対して行うため exp をかける
    log_alpha = noisy_logits / tau
    input_matrix = torch.exp(log_alpha)

    # 3. Sinkhorn 反復 (行と列の正規化を繰り返す)
    # これにより、全ての行和と列和を 1.0 に近づける
    curr_matrix = input_matrix
    
    for _ in range(n_iters):
        # 行(dim=2)方向の正規化
        curr_matrix = curr_matrix / (torch.sum(curr_matrix, dim=2, keepdim=True) + 1e-6)
        # 列(dim=1)方向の正規化
        curr_matrix = curr_matrix / (torch.sum(curr_matrix, dim=1, keepdim=True) + 1e-6)

    return curr_matrix
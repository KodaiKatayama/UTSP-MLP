import torch
from scipy.optimize import linear_sum_assignment

def get_adjacency_matrix(distance_matrix, scale):
    adjacency_matrix = torch.exp(-distance_matrix / scale)
    return adjacency_matrix

def sample_gumbel(shape):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u))

def sinkhorn(logit, n_iters):
    T = torch.exp(logit)
    for _ in range(n_iters):
        T = T / torch.sum(T, dim=-1, keepdim=True)
        T = T / torch.sum(T, dim=-2, keepdim=True)
    return T

def gumbel_sinkhorn(logit, tau, gamma, n_iters):
    gumbel_noise = sample_gumbel(logit.shape)
    noisy_logits = (logit + gamma * gumbel_noise) / tau
    return sinkhorn(noisy_logits, n_iters)

def Hungarian(logit):
    logit_np = logit.detach().cpu().numpy()
    row, col = linear_sum_assignment(logit_np, maximize=True)
    P = torch.zeros_like(logit)
    P[row, col] = 1
    return P
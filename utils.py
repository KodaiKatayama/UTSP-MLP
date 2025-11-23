import torch

def get_adjacency_matrix(distance_matrix, scale):
    adjacency_matrix = torch.exp(-distance_matrix / scale)
    return adjacency_matrix

def sample_gumbel(shape):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u))

def sinkhorn(logits, n_iters):
    P = torch.exp(logits)
    for _ in range(n_iters):
        P = P / torch.sum(P, dim=-1, keepdim=True)
        P = P / torch.sum(P, dim=-2, keepdim=True)
    return P
    
import torch

def get_adjacency_matrix(distance_matrix, scale):
    adjacency_matrix = torch.exp(-distance_matrix / scale)
    return adjacency_matrix
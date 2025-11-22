import torch
from torch.utils.data import Dataset

class TSPDataset(Dataset):
    def __init__(self, num_samples, num_nodes, seed=None):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.points = torch.rand(num_samples, num_nodes, 2)
        self.distances = torch.cdist(self.points, self.points, p=2)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return {
            'points': self.points[idx],
            'distance': self.distances[idx]
        }

if __name__ == "__main__":
    dataset = TSPDataset(num_samples=2, num_nodes=4, seed=42)

    print("--- 都市の座標 ---")
    print(dataset.points[0])
    print(dataset.points[1])
    
    print("\n--- 距離行列 ---")
    print(dataset.distances[0])
    print(dataset.distances[1])
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh


seed_everything(42)

dataset = KarateClub()

print(f'Dataset Size: {len(dataset)}')
print(f'Number of Classes: {dataset.num_classes}')
print(f'Number of Node Features: {dataset.num_node_features}')
print()

data = dataset[0]

print(data)

edge_index, edge_weight = get_laplacian(data.edge_index, 
                                        normalization='sym')

n = data.num_nodes
laplacian = torch.zeros(size=(n, n))
laplacian[edge_index[0, :], edge_index[1, :]] = edge_weight

K = 3
eigenvalues, eigenvectors = eigsh(laplacian.numpy(), k=K, which='LM')
eigenvectors = torch.from_numpy(eigenvectors)
out = eigenvectors.T @ data.x

out_channels = 9
spec = torch.nn.Parameter(torch.randn(out_channels, K, n))
out = spec * out
out = out.sum(dim=-1, keepdim=True)
out = eigenvectors @ out
out = out.squeeze(-1)
out = F.softmax(out, dim=-1).T

print(out)
print(out.shape)

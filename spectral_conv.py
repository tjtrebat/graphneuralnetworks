import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh

dataset = KarateClub()

print(f'Dataset Size: {len(dataset)}')
print(f'Number of Classes: {dataset.num_classes}')
print(f'Number of Node Features: {dataset.num_node_features}')
print()

data = dataset[0]

print(data)

print(f'Is Undirected?: {data.is_undirected()}')
print(f'Nodes to Train: {data.train_mask.sum().item()}')

edge_index, edge_weight = get_laplacian(data.edge_index, 
                                        normalization='sym')

laplacian = torch.zeros(size=(data.num_nodes, data.num_nodes))
laplacian[edge_index[0, :], edge_index[1, :]] = edge_weight

print(laplacian)
print(torch.diag(laplacian))

eigenvalues, eigenvectors = eigsh(laplacian.numpy(), k=3, which='LM')

print(eigenvectors.shape)

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh

from models import GCN, ChebNet, SpecNet


seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = KarateClub()
print(f'Dataset Size: {len(dataset)}')
print(f'Number of Classes: {dataset.num_classes}')
print(f'Number of Node Features: {dataset.num_node_features}')
print()

data = dataset[0]
print(data)



def get_laplacian_matrix(edge_index, num_nodes):
    edge_index, edge_weight = get_laplacian(edge_index, 
                                            normalization='sym')

    laplacian = torch.zeros(size=(num_nodes,) * 2)
    laplacian[edge_index[0, :], edge_index[1, :]] = edge_weight
    return laplacian

def get_eigendecomposition(laplacian, K=10):
    _, U = eigsh(laplacian.numpy(), k=K, which='LM')
    return torch.from_numpy(U)

def evaluate(model, data):        
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.01, 
                                 weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred == data.y).sum()
    acc = int(correct) / len(data.y)
    return acc  



laplacian = get_laplacian_matrix(data.edge_index, data.num_nodes)
U = get_eigendecomposition(laplacian)
U = U.to(device)

data = data.to(device)

model = GCN(dataset.num_node_features, dataset.num_classes)
model = model.to(device)
acc = evaluate(model, data)
print(f'Accuracy (GCN): {acc:.4f}')
print()

for k in range(1, 10):
    model = ChebNet(dataset.num_node_features, dataset.num_classes, K=k)
    model = model.to(device)
    acc = evaluate(model, data)
    print(f'Accuracy (ChebNet (K={k}): {acc:.4f}')    

print()

for k in range(1, 10):
    model = SpecNet(dataset.num_node_features, dataset.num_classes, U[:, :k])
    model = model.to(device)
    acc = evaluate(model, data)
    print(f'Accuracy (SpecNet (K={k}): {acc:.4f}')    

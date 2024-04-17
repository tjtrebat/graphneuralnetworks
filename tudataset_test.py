import argparse

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh

from models import GCN, ChebNet, SpecNet


seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=['GCN', 'ChebNet', 'SpecNet'],
                        help='model to train/evaluate')
    parser.add_argument('--k',
                        type=int,
                        default=3,
                        help='number of eigenvectors/polynomials')
    parser.add_argument('--dataset',
                        choices=['IMDB-BINARY', 'ENZYMES'],
                        help='dataset to train and evaluate models')    
    return parser

parser = get_parser()
args = parser.parse_args()
dataset_name = args.dataset
dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name, use_node_attr=True)

def print_dataset_stats(dataset):
    print(dataset)
    print(f'Dataset Size: {len(dataset)}')
    print(f'Number of Classes: {dataset.num_classes}')
    print(f'Number of Node Features: {dataset.num_node_features}')

print_dataset_stats(dataset)

def get_laplacian_matrix(edge_index, num_nodes):
    edge_index, edge_weight = get_laplacian(edge_index, 
                                            normalization='sym')

    laplacian = torch.zeros(size=(num_nodes,) * 2)
    laplacian[edge_index[0, :], edge_index[1, :]] = edge_weight
    return laplacian

def get_eigendecomposition(laplacian, k):
    _, U = eigsh(laplacian.numpy(), k=k, which='LM')
    return torch.from_numpy(U)

def get_specnet(dataset, k):
    data = dataset[0]
    laplacian = get_laplacian_matrix(data.edge_index, data.num_nodes)
    eigen = get_eigendecomposition(laplacian, k).to(device)
    model = SpecNet(dataset.num_node_features, dataset.num_classes, eigen)
    return model

def get_model(model_name, dataset, k):
    if model_name == 'GCN':
        model = GCN(dataset.num_node_features, dataset.num_classes)
    elif model_name == 'ChebNet':
        model = ChebNet(dataset.num_node_features, dataset.num_classes, K=k)
    else:
        model = get_specnet(dataset, k)
    return model

model = get_model(args.model, dataset, args.k)
model = model.to(device)

train_dataset, test_dataset = dataset[:540], dataset[540:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


model.train()
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.01, 
                                weight_decay=5e-4)
for epoch in range(200):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = F.nll_loss(out, data.y[data.batch])
        loss.backward()
        optimizer.step()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()
correct = 0
total = 0
for data in test_loader:
    pred = model(data.to(device)).argmax(dim=1)
    print(pred)
    print(data.y[data.batch])
    correct += (pred == data.y[data.batch]).sum()
    total += len(data)
    #print(correct)
    #print(total)

acc = correct / total
print(f'Accuracy: {acc:.4f}') 

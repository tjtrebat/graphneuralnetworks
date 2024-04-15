import argparse

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
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
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='dataset to train and evaluate models')    
    return parser

parser = get_parser()
args = parser.parse_args()
dataset_name = args.dataset
dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
data = dataset[0].to(device)

def print_dataset_stats(dataset):
    print(dataset)
    print(f'Dataset Size: {len(dataset)}')
    print(f'Number of Classes: {dataset.num_classes}')
    print(f'Number of Node Features: {dataset.num_node_features}')
    data = dataset[0]
    print(f'Is Undirected?: {data.is_undirected()}')
    print(f'Nodes to Train: {data.train_mask.sum().item()}')
    print(f'Validation Nodes: {data.val_mask.sum().item()}')
    print(f'Test Nodes: {data.test_mask.sum().item()}')

print_dataset_stats(dataset)
print(data)

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
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc   

acc = evaluate(model, data)
model_name = args.model
if model_name != 'GCN':
    model_name += f' (K={args.k})'
print(f'Accuracy ({model_name}): {acc:.4f}')

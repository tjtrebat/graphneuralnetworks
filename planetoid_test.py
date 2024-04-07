import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models import GCN, ChebNet


dataset_names = ['Cora', 'CiteSeer', 'PubMed']
datasets = [Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name) 
            for dataset_name in dataset_names]

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
 

for dataset in datasets:
    print_dataset_stats(dataset)
    data = dataset[0].to(device)

    models = {
        'GCN': GCN(dataset.num_node_features, 
                   dataset.num_classes),
        'ChebNet (K=1)': ChebNet(dataset.num_node_features, 
                                 dataset.num_classes),
        'ChebNet (K=2)': ChebNet(dataset.num_node_features, 
                                 dataset.num_classes, 
                                 K=2),
        'ChebNet (K=3)': ChebNet(dataset.num_node_features, 
                                 dataset.num_classes, 
                                 K=3)}
    for model_name, model in models.items():
        model = model.to(device)
        acc = evaluate(model, data)
        print(f'Accuracy ({model_name}): {acc:.4f}')
    print()


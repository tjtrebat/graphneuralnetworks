import time
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr
from torch_geometric.logging import log

from models import GCN, ChebNet


parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['GCN', 'ChebNet'],
                    help='model to train/evaluate')
parser.add_argument('--k',
                    type=int,
                    default=2,
                    help='number of eigenvectors/polynomials')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Flickr(root='/tmp/Flickr', transform=T.NormalizeFeatures())
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

if args.model == 'GCN':
    model = GCN(dataset.num_node_features, 
                dataset.num_classes).to(device)
else:
    model = ChebNet(dataset.num_node_features, 
                    dataset.num_classes, 
                    K=args.k).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01, 
                             weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(data).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

best_val_acc = test_acc = 0
times = []
for epoch in range(1, 200 + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

import time
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.logging import log
from torch_geometric import seed_everything
from torch_geometric.datasets import Flickr

from models import model_factory


seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['GCN', 'ChebNet'],
                    help='model to train/evaluate')
parser.add_argument('--k',
                    type=int,
                    default=1,
                    help='number of eigenvectors/polynomials')
parser.add_argument('--dataset',
                    choices=['Flickr'],
                    help='dataset to train and evaluate models')

args = parser.parse_args()

dataset = Flickr(root='/tmp/Flickr', transform=T.NormalizeFeatures())
print(dataset)
print(f'Dataset Size: {len(dataset)}')
print(f'Number of Classes: {dataset.num_classes}')
print(f'Number of Node Features: {dataset.num_node_features}')
data = dataset[0].to(device)
print(f'Is Undirected?: {data.is_undirected()}')
print(f'Nodes to Train: {data.train_mask.sum().item()}')
print(f'Validation Nodes: {data.val_mask.sum().item()}')
print(f'Test Nodes: {data.test_mask.sum().item()}')

model = model_factory.create(args.model, dataset=dataset, k=args.k).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01, 
                             weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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

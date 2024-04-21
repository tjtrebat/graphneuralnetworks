import time
import argparse

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.datasets import KarateClub
from torch_geometric.logging import log

from model_factory import model_factory


seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['GCN', 'ChebNet', 'SpecNet'],
                    help='model to train/evaluate')
parser.add_argument('--k',
                    type=int,
                    default=3,
                    help='number of eigenvectors/polynomials')

args = parser.parse_args()

dataset = KarateClub()
print(f'Dataset Size: {len(dataset)}')
print(f'Number of Classes: {dataset.num_classes}')
print(f'Number of Node Features: {dataset.num_node_features}')
print(dataset[0])

data = dataset[0].to(device)
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
    for mask in [data.train_mask, torch.logical_not(data.train_mask)]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

test_acc = 0
times = []
for epoch in range(1, 200 + 1):
    start = time.time()
    loss = train()
    train_acc, tmp_test_acc = test()
    if tmp_test_acc > test_acc:
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

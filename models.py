import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GCNConv, ChebConv

from spec_conv import SpecConv


class GCN(torch.nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_feats, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, n_feats, n_classes, K=1):
        super().__init__()
        self.conv1 = ChebConv(n_feats, 16, K)
        self.conv2 = ChebConv(16, n_classes, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        

class SpecNet(torch.nn.Module):
    def __init__(self, n_feats: int, n_classes: int, U: Tensor):
        super().__init__()
        self.conv1 = SpecConv(n_feats, 16, U)
        self.conv2 = SpecConv(16, n_classes, U)

    def forward(self, data):
        x = data.x
        x = self.conv1(x)    
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)
    
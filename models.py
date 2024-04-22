import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import get_laplacian
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
    
    
class ModelFactory:
    def __init__(self):
        self._models = {}

    def register_model(self, key, model):
        self._models[key] = model

    def create(self, key, **kwargs):
        model = self._models[key]
        if not model:
            raise ValueError(key)
        return model(**kwargs)
    

def create_gcn_model(dataset, **_ignored):
    return GCN(dataset.num_node_features, dataset.num_classes)

def create_chebnet_model(dataset, k, **_ignored):
    return ChebNet(dataset.num_node_features, dataset.num_classes, K=k)

def get_laplacian_matrix(edge_index, num_nodes):
    edge_index, edge_weight = get_laplacian(edge_index, 
                                            normalization='sym')

    laplacian = torch.zeros(size=(num_nodes,) * 2)
    laplacian[edge_index[0, :], edge_index[1, :]] = edge_weight
    return laplacian

def get_eigendecomposition(laplacian, k):
    _, U = eigsh(laplacian.numpy(), k=k, which='LM')
    return torch.from_numpy(U)

def create_specnet_model(dataset, k, **_ignored):
    data = dataset[0]
    laplacian = get_laplacian_matrix(data.edge_index, data.num_nodes)
    U = get_eigendecomposition(laplacian, k)
    model = SpecNet(dataset.num_node_features, dataset.num_classes, U)
    return model

model_factory = ModelFactory()
model_factory.register_model('GCN', create_gcn_model)
model_factory.register_model('ChebNet', create_chebnet_model)
model_factory.register_model('SpecNet', create_specnet_model)

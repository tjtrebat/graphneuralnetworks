import torch
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh

from models import GCN, ChebNet, SpecNet

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
    eigen = get_eigendecomposition(laplacian, k)
    model = SpecNet(dataset.num_node_features, dataset.num_classes, eigen)
    return model

model_factory = ModelFactory()
model_factory.register_model('GCN', create_gcn_model)
model_factory.register_model('ChebNet', create_chebnet_model)
model_factory.register_model('SpecNet', create_specnet_model)

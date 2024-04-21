import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear


class SpecConv(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels: int, 
                 U: Tensor):
        super().__init__()
        self.U = U
        self.lin = Linear(in_channels, 
                          out_channels, 
                          bias=False, 
                          weight_initializer='glorot')
        
    def forward(self, x):
        out = self.U.T @ x
        out = self.lin(out)
        out = self.U @ out
        return out
        

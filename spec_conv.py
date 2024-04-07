import torch
import torch.nn.functional as F
from torch import Tensor


class SpecConv(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels: int, 
                 U: Tensor):
        super().__init__()
        self.U = U
        self.coeffs = torch.nn.Parameter(
            torch.randn(out_channels, U.size(-1), in_channels))
        
    def forward(self, x):
        out = self.U.T @ x
        out = self.coeffs * out
        out = out.sum(dim=-1, keepdim=True)
        out = self.U @ out
        out = out.squeeze(-1)
        out = F.softmax(out, dim=-1).T
        return out
        

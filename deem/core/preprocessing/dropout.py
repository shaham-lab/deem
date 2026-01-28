"""Dropout layer for last dimension.

Contains Dropout1d applied to the last dimension of a tensor.
"""

import torch
import torch.nn as nn


class Dropout1dLastDim(nn.Module):
    """Apply Dropout1d to the last dimension of a tensor.
    
    Permutes the tensor to apply nn.Dropout1d along the last dimension,
    then permutes back to the original shape.
    
    Args:
        p: Probability of an element to be zeroed. Default: 0.5
        
    Example:
        >>> dropout = Dropout1dLastDim(p=0.2)
        >>> x = torch.randn(32, 10, 64)  # (N, L, C)
        >>> output = dropout(x)
        >>> assert output.shape == x.shape
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout1d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute the tensor to move the last dimension to the second position
        x_permuted = x.permute(0, 2, 1)  # N x L x C -> N x C x L

        # Apply Dropout1d
        x_dropped = self.dropout(x_permuted)

        # Permute the tensor back to its original shape
        x_result = x_dropped.permute(0, 2, 1)  # N x C x L -> N x L x C

        return x_result

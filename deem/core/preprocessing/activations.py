"""Activation functions for preprocessing layers.

Contains sparse activation functions like Sparsemax, as well as
experimental activation functions like SinSoftmax.
"""

import numpy as np
import torch
import torch.nn as nn

from .utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim


class Sparsemax(nn.Module):
    """Sparsemax activation function.
    
    Projects input onto the probability simplex, producing sparse outputs.
    Unlike softmax which produces dense probability distributions, sparsemax
    assigns exactly zero probability to some outputs.
    
    Reference: https://arxiv.org/pdf/1602.02068.pdf
    
    Args:
        dim: The dimension over which to apply sparsemax. Default: -1
        
    Example:
        >>> sparsemax = Sparsemax(dim=-1)
        >>> x = torch.randn(32, 10)
        >>> output = sparsemax(x)
        >>> # output sums to 1 along dim=-1, with some entries exactly 0
    """
    __constants__ = ["dim"]

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SparsemaxFunction.apply(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class SparsemaxFunction(torch.autograd.Function):
    """Autograd function for sparsemax.
    
    Implements both forward and backward pass for the sparsemax operation.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1) -> torch.Tensor:
        input_dim = input.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, input = flatten_all_but_nth_dim(ctx, input)

        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range_tensor = torch.arange(1, input.size()[-1] + 1)
        range_tensor = range_tensor.expand_as(input).to(input)

        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(input.dtype)
        k = (is_gt * range_tensor).max(-1, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        sum_grad = (grad_output * nonzeros).sum(-1, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum_grad.expand_as(grad_output))

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None


class Sin2maxShifted(nn.Module):
    """Shifted sin-squared based activation that produces probability-like outputs.
    
    Applies sin(x + π/4)² element-wise and normalizes to sum to 1.
    About 95 ms for (64, 8, 196, 196) on CPU.
    
    Args:
        dim: Dimension along which to normalize. Default: -1
        prenorm: Whether to apply prenormalization. Default: False
        eps: Small constant for numerical stability. Default: 1e-05
    """
    
    def __init__(self, dim: int = -1, prenorm: bool = False, eps: float = 1e-05):
        super().__init__()
        self.dim = dim
        self.prenorm = prenorm
        self.eps = eps
        self.pre = nn.Identity()
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor + 0.25 * np.pi) ** 2
        sum_term = input_tensor.sum(dim=self.dim, keepdim=True)
        
        # Repeat the summed tensor along the same dimension
        sum_term = sum_term.repeat_interleave(
            input_tensor.shape[self.dim], dim=self.dim
        )
        output = input_tensor / sum_term
        return output


class SinSoftmax(nn.Module):
    """Sin-based softmax activation.
    
    Applies sin() element-wise followed by softmax normalization.
    About 55 ms for (64, 8, 196, 196) on CPU.
    
    Args:
        dim: Dimension along which to apply softmax. Default: -1
        prenorm: Whether to apply prenormalization. Default: False
        eps: Small constant for numerical stability. Default: 1e-05
    """
    
    def __init__(self, dim: int = -1, prenorm: bool = False, eps: float = 1e-05):
        super().__init__()
        self.dim = dim
        self.prenorm = prenorm
        self.eps = eps
        self.pre = nn.Identity()
        self.softmax = nn.Softmax(dim=self.dim)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = self.pre(input_tensor)
        input_tensor = torch.sin(input_tensor)
        return self.softmax(input_tensor)

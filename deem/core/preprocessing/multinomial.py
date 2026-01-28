"""Multinomial transformation layer for RBM preprocessing.

This module provides the Multinomial layer which applies a learnable
transformation to multinomial-structured inputs (e.g., classifier predictions).
"""

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from .activations import Sparsemax, SinSoftmax
from .dropout import Dropout1dLastDim


# Try to import entmax functions, fall back to sparsemax if not available
try:
    from entmax import entmax15, sparsemax as entmax_sparsemax, entmax_bisect
    HAS_ENTMAX = True
except ImportError:
    HAS_ENTMAX = False


def _safe_einsum(equation: str, *operands) -> Tensor:
    """Simple einsum wrapper (can be replaced with memory-safe version if needed)."""
    return torch.einsum(equation, *operands)


class Multinomial(nn.Module):
    """Multinomial transformation layer.
    
    Applies a learned linear transformation to multinomial-structured data,
    mapping from (batch, in_multi_units, in_features) to (batch, out_multi_units, out_features).
    
    This layer is useful for preprocessing classifier ensemble predictions before
    feeding them to an RBM. It can learn to transform the raw predictions into
    a more suitable representation.
    
    Args:
        in_multi_units: Number of input multinomial units (e.g., number of classifiers)
        out_multi_units: Number of output multinomial units
        in_features: Size of each input feature (e.g., number of classes)
        out_features: Size of each output feature
        init_method: Weight initialization method. Options:
            - 'default': Kaiming uniform initialization
            - 'rand': Random normal initialization
            - 'identity': Identity initialization (requires matching dims)
            - 'mv': Majority vote initialization
        one_hot: If True, output one-hot vectors using hard Gumbel-Softmax
        use_softmax: If True, apply activation function to outputs
        dbn_mode: If True, use DBN-style bias (for hidden-to-visible)
        device: Device for parameters
        dtype: Data type for parameters
        jitter_coef: Coefficient for jittering weights (used with identity init)
        activation_func_name: Activation function to use. Options:
            - 'sparsemax': Sparsemax activation (default)
            - 'entmax': Entmax-1.5 activation
            - 'ultramax': Entmax with alpha=3
            - 'softmax': Standard softmax
            - 'gelu': GELU activation
            - 'silu': SiLU/Swish activation
            
    Shape:
        - Input: (N, in_multi_units, in_features)
        - Output: (N, out_multi_units, out_features)
        
    Example:
        >>> layer = Multinomial(20, 30, 40, 50)
        >>> x = torch.randn(128, 20, 40)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([128, 30, 50])
    """
    
    __constants__ = ['in_features', 'out_features', 'in_multi_units', 'out_multi_units']
    in_features: int
    out_features: int
    in_multi_units: int
    out_multi_units: int
    weights: Tensor

    InitMethod = Literal['default', 'rand', 'identity', 'mv']
    ActivationName = Literal['sparsemax', 'entmax', 'ultramax', 'softmax', 'gelu', 'silu']

    def __init__(
        self,
        in_multi_units: int,
        out_multi_units: int,
        in_features: int,
        out_features: int,
        init_method: str = 'default',
        one_hot: bool = False,
        use_softmax: bool = False,
        dbn_mode: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        jitter_coef: float = 0.005,
        activation_func_name: str = 'sparsemax',
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.in_multi_units = in_multi_units
        self.out_multi_units = out_multi_units
        self.one_hot = one_hot
        self.use_softmax = use_softmax
        self.jitter_coef = jitter_coef
        self.activation_func_name = activation_func_name
        self.dbn_mode = dbn_mode
        
        # Initialize weights
        self.weights = Parameter(torch.empty(
            (in_multi_units, out_multi_units, in_features, out_features),
            **factory_kwargs
        ))
        
        # Initialize bias (different shape for DBN mode)
        if dbn_mode:
            self.bias = Parameter(torch.empty(
                (in_multi_units, in_features), **factory_kwargs
            ))
        else:
            self.bias = Parameter(torch.empty(
                (out_multi_units, out_features), **factory_kwargs
            ))
        
        # Set up activation function
        self._setup_activation_func(activation_func_name)
        
        # Initialize parameters based on method
        if init_method == 'rand':
            self.init_randn()
        elif init_method == 'identity':
            self.init_identity()
        elif init_method == 'mv':
            self.init_majority_vote()
        else:  # default, same as nn.Linear
            self.reset_parameters()
        
        # Additional layers
        self.sparsemax = Sparsemax(dim=1)
        self.sin_softmax = SinSoftmax(dim=1)
        self.dropout = Dropout1dLastDim(0.2)
    
    def _setup_activation_func(self, name: str) -> None:
        """Set up the activation function based on name."""
        if name == 'gelu':
            self.activation_func = nn.GELU()
        elif name == 'silu':
            self.activation_func = nn.SiLU()
        elif name == 'softmax':
            self.activation_func = nn.Softmax(dim=1)
        else:
            self.activation_func = Sparsemax(dim=1)
    
    def init_randn(self) -> None:
        """Initialize weights and bias with random normal distribution."""
        init.normal_(self.weights)
        init.normal_(self.bias)

    @torch.no_grad()
    def init_identity(self) -> None:
        """Initialize as identity transformation.
        
        Requires matching input/output dimensions.
        
        Raises:
            ValueError: If dimensions don't match for identity initialization.
        """
        if self.out_features != self.in_features or self.out_multi_units != self.in_multi_units:
            raise ValueError(
                "Identity initialization requires matching dimensions: "
                f"in=({self.in_multi_units}, {self.in_features}), "
                f"out=({self.out_multi_units}, {self.out_features})"
            )

        init.zeros_(self.weights)
        init.zeros_(self.bias)
        
        for i in range(self.in_multi_units):
            for j in range(self.in_features):
                self.weights[i, i, j, j] = 1.0
        
        self.jitter_weights()
    
    @torch.no_grad()
    def init_majority_vote(self) -> None:
        """Initialize for majority vote aggregation.
        
        Requires matching multinomial dimensions.
        
        Raises:
            ValueError: If multinomial dimensions don't match.
        """
        if self.out_multi_units != self.in_multi_units:
            raise ValueError(
                "Majority vote initialization requires matching multinomial dims: "
                f"in={self.in_multi_units}, out={self.out_multi_units}"
            )

        init.zeros_(self.weights)
        init.zeros_(self.bias)

        for i in range(self.in_multi_units):
            self.weights[i][i] = torch.ones(
                self.in_features, self.out_features
            ) * 2

    @torch.no_grad()
    def jitter_weights(self, jitter_coef: Optional[float] = None) -> None:
        """Add small random noise to weights and bias.
        
        Args:
            jitter_coef: Noise coefficient. If None, uses self.jitter_coef.
        """
        coef = jitter_coef if jitter_coef is not None else self.jitter_coef
        self.weights += torch.randn_like(self.weights) * coef
        self.bias += torch.randn_like(self.bias) * coef

    def reset_parameters(self) -> None:
        """Initialize using Kaiming uniform (same as nn.Linear default)."""
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, temperature: float = 1.0) -> Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor of shape (N, in_multi_units, in_features)
            temperature: Temperature for activation function scaling
            
        Returns:
            Output tensor of shape (N, out_multi_units, out_features)
        """
        logits = _safe_einsum('lmij,nli->nmj', self.weights, input) + self.bias
        
        if not self.use_softmax:
            return logits
            
        if self.one_hot:
            return F.gumbel_softmax(logits, tau=0.1, hard=True, dim=1)
        
        # Apply activation function based on name
        if self.activation_func_name == 'ultramax':
            if not HAS_ENTMAX:
                raise ImportError("entmax package required for ultramax activation")
            return entmax_bisect(logits / temperature, alpha=3, dim=1)
        elif self.activation_func_name == 'entmax':
            if not HAS_ENTMAX:
                raise ImportError("entmax package required for entmax activation")
            return entmax15(logits / temperature, dim=1)
        elif self.activation_func_name == 'sparsemax':
            if HAS_ENTMAX:
                return entmax_sparsemax(logits / temperature, dim=1)
            return self.sparsemax(logits / temperature)
        
        return self.activation_func(logits / temperature)
    
    def forward_dbn(self, input: Tensor, bias: Tensor, temperature: float = 1.0) -> Tensor:
        """Forward pass for Deep Belief Network mode.
        
        Args:
            input: Input tensor
            bias: External bias to use instead of self.bias
            temperature: Temperature for activation function scaling
            
        Returns:
            Output tensor
        """
        logits = _safe_einsum('lmij,nli->nmj', self.weights, input) + bias
        
        if self.one_hot:
            return F.gumbel_softmax(logits * 5, tau=1, hard=True, dim=1)
        
        return self.sparsemax(logits / temperature)
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'l={self.in_multi_units}, m={self.out_multi_units}, '
            f'bias={self.bias is not None}'
        )

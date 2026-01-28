"""Preprocessing layers for RBM models.

This module contains optional preprocessing layers that can be applied
before RBM training, including the Multinomial transformation layer
and various activation functions.
"""

from .multinomial import Multinomial
from .activations import (
    Sparsemax, 
    SparsemaxFunction,
    SinSoftmax,
    Sin2maxShifted,
)
from .dropout import Dropout1dLastDim

__all__ = [
    'Multinomial',
    'Sparsemax',
    'SparsemaxFunction', 
    'SinSoftmax',
    'Sin2maxShifted',
    'Dropout1dLastDim',
]

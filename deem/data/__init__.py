"""Data loading utilities for deem package.

This module provides utilities for loading ensemble prediction data:
- Hard labels: (N, D) integer arrays with classifier predictions
- Soft labels: (N, K, D) probability distributions

Example usage:
    >>> from deem.data import load_dataset
    >>> train, val, test, meta = load_dataset('data.mat')
    >>> print(meta['format'])  # 'hard_labels' or 'soft_labels'
"""

from .loaders import (
    load_dataset,
    load_from_mat,
    load_from_mat_soft
)
from .datasets import TensorDatasetWithShape
from .transforms import (
    convert_soft_to_hard,
    filter_missing_samples
)

__all__ = [
    'load_dataset',
    'load_from_mat',
    'load_from_mat_soft',
    'TensorDatasetWithShape',
    'convert_soft_to_hard',
    'filter_missing_samples'
]

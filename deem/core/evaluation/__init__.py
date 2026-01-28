"""Evaluation utilities for RBM models.

This module provides tools for evaluating RBM predictions, including:
- Hungarian algorithm for solving the label permutation problem
- Accuracy metrics with proper label alignment
- Dead class detection and analysis
"""

from .hungarian import (
    get_hungarian_solution,
    vectorize_predictions,
    build_cost_matrix,
)
from .metrics import (
    accuracy_with_hungarian,
    compute_mean_std,
    extract_dead_classes_indices,
)

__all__ = [
    'get_hungarian_solution',
    'vectorize_predictions',
    'build_cost_matrix',
    'accuracy_with_hungarian',
    'compute_mean_std',
    'extract_dead_classes_indices',
]

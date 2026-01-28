"""Training utilities for RBM models.

This module provides the training loop and related utilities for training
Restricted Boltzmann Machines using contrastive divergence.

Key Components
--------------
RBMTrainer
    Production training loop extracted from run_predict.py.
    Handles all batches without the legacy break bug.

apply_l1_regularization
    L1 regularization utility for weight sparsity.

TrainingCallback (Protocol)
    Interface for custom training callbacks.
"""

from .trainer import RBMTrainer, TrainingCallback, apply_l1_regularization

__all__ = ['RBMTrainer', 'TrainingCallback', 'apply_l1_regularization']

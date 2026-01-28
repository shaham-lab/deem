"""Loss functions for RBM training.

This module provides energy-based loss functions for training
Restricted Boltzmann Machines using contrastive divergence.
"""

from .ebm_loss import EBMLoss, ContrastiveDivergenceLoss

__all__ = ['EBMLoss', 'ContrastiveDivergenceLoss']

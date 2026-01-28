"""RBM model implementations for deem package.

This module provides base classes for Restricted Boltzmann Machines:
- RBM: Abstract base class defining the interface
- MultinomialRBM: Mixin with multinomial-specific functionality
- MultinomialRBMGwg: Production model with DLP/GWG sampling

Example usage:
    >>> from deem.core.models import MultinomialRBMGwg
    >>> rbm = MultinomialRBMGwg(dx=15, dh=3, k=3, l=3, m=3, device='cpu')
    >>> predictions = rbm.predict(data)
"""

from .base import RBM, MultinomialRBM
from .rbm_gwg import MultinomialRBMGwg, RBMGwg

__all__ = ['RBM', 'MultinomialRBM', 'MultinomialRBMGwg', 'RBMGwg']

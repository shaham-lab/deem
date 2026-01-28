"""
DEEM - Deep Ensemble Energy Models

A library for training Restricted Boltzmann Machines (RBMs) on ensemble predictions
from multiple classifiers (crowd learning, model aggregation).

Example
-------
>>> from deem import DEEM
>>> import numpy as np
>>>
>>> predictions = np.random.randint(0, 3, (100, 15))
>>> model = DEEM()
>>> model.fit(predictions)
>>> consensus = model.predict(predictions)
"""

__version__ = "0.2.0"

from . import data
from . import core
from . import automl

# High-level API
from .ensemble import DEEM

__all__ = ['DEEM', 'data', 'core', 'automl']

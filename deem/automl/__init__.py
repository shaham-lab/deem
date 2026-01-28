"""AutoML components for hyperparameter prediction.

This module provides:
- MetaFeatureExtractor: Extract 28 meta-features from ensemble prediction data
- HyperparameterPredictor: Predict optimal hyperparameters from meta-features

Example usage:
    >>> from deem.automl import HyperparameterPredictor
    >>> predictor = HyperparameterPredictor(model_dir='saved_hyp_models_v1')
    >>> hyperparams = predictor.predict(predictions)
    >>> print(hyperparams['learning_rate'], hyperparams['batch_size'])
"""

from .meta_features import MetaFeatureExtractor
from .hyperparameters import HyperparameterPredictor

__all__ = [
    'MetaFeatureExtractor',
    'HyperparameterPredictor',
]

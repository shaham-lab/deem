"""Hyperparameter prediction using trained sklearn models.

This module provides the HyperparameterPredictor class that uses pre-trained
sklearn models to predict optimal hyperparameters for RBM training based on
dataset meta-features.

The predictor loads models from saved_hyp_models_v1/ directory and predicts
8 hyperparameters:
- batch_size
- epochs
- learning_rate (lr)
- init_method
- num_layers
- activation_func
- momentum
- scheduler
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .meta_features import FEATURE_NAMES, MetaFeatureExtractor

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Default hyperparameters if prediction fails
DEFAULT_HYPERPARAMETERS = {
    'batch_size': 2048,
    'epochs': 50,
    'learning_rate': 0.001,
    'init_method': 'mv_rand',
    'num_layers': 1,
    'activation_func': 'sparsemax',
    'momentum': 0.0,
    'scheduler': 'None',
}

# Mapping from model file names to hyperparameter keys
PARAM_FILE_MAPPING = {
    'param_batch_size_clf.pkl': 'batch_size',
    'param_epochs_clf.pkl': 'epochs',
    'param_lr_clf.pkl': 'learning_rate',
    'param_init_method_clf.pkl': 'init_method',
    'param_num_layers_clf.pkl': 'num_layers',
    'param_activation_func_clf.pkl': 'activation_func',
    'param_momentum_clf.pkl': 'momentum',
    'param_scheduler_clf.pkl': 'scheduler',
}

# Order of hyperparameters as used in update_config_by_hyperparameters()
# [batch_size, epochs, lr, init_method, num_layers, activation_func, momentum, scheduler]
HYPERPARAM_ORDER = [
    'batch_size', 'epochs', 'learning_rate', 'init_method',
    'num_layers', 'activation_func', 'momentum', 'scheduler'
]


class HyperparameterPredictor:
    """Predict optimal hyperparameters from dataset meta-features.
    
    Uses trained sklearn models to predict 8 hyperparameters:
    - batch_size: Training batch size
    - epochs: Number of training epochs
    - learning_rate: Learning rate for optimizer
    - init_method: Weight initialization method
    - num_layers: Number of preprocessing layers
    - activation_func: Activation function for preprocessing
    - momentum: Optimizer momentum
    - scheduler: Learning rate scheduler
    
    Parameters
    ----------
    model_dir : str or Path or None, default=None
        Directory containing trained sklearn model files (*.pkl).
        If None (default), uses bundled models from the installed package.
        Specify a custom path to use external models (e.g., for experiments).
    raise_on_missing : bool, default=False
        If True, raise error when model files are missing.
        If False, use default hyperparameters for missing models.
        
    Examples
    --------
    >>> import numpy as np
    >>> from deem.automl import HyperparameterPredictor
    >>> 
    >>> # Use bundled models (default - works out-of-the-box!)
    >>> predictor = HyperparameterPredictor()
    >>> 
    >>> # Or use custom models
    >>> predictor = HyperparameterPredictor(model_dir='my_models/')
    >>> 
    >>> # Predict hyperparameters from data
    >>> predictions = np.random.randint(0, 3, size=(1000, 15))
    >>> hyperparams = predictor.predict(predictions)
    >>> print(hyperparams['learning_rate'], hyperparams['batch_size'])
    """
    
    def __init__(
        self,
        model_dir: str | Path | None = None,
        raise_on_missing: bool = False,
    ):
        # Default to packaged models if not specified
        if model_dir is None:
            # Use packaged models from deem/automl/models/
            import importlib.resources
            try:
                # Python 3.9+ style
                if hasattr(importlib.resources, 'files'):
                    model_dir = importlib.resources.files('deem.automl.models')
                else:
                    # Python 3.8 fallback
                    with importlib.resources.path('deem.automl', 'models') as path:
                        model_dir = path
            except Exception:
                # Final fallback - try relative path
                model_dir = Path(__file__).parent / 'models'
        
        self.model_dir = Path(model_dir)
        self.raise_on_missing = raise_on_missing
        self.models: dict[str, Any] = {}
        self.feature_extractor = MetaFeatureExtractor()
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained sklearn models from model directory."""
        if not self.model_dir.exists():
            if self.raise_on_missing:
                raise FileNotFoundError(
                    f"Model directory not found: {self.model_dir}"
                )
            warnings.warn(
                f"Model directory not found: {self.model_dir}. "
                "Will use default hyperparameters.",
                UserWarning,
            )
            return
        
        try:
            import joblib
        except ImportError:
            if self.raise_on_missing:
                raise ImportError("joblib is required for loading models")
            warnings.warn(
                "joblib not installed. Will use default hyperparameters.",
                UserWarning,
            )
            return
        
        for filename, param_key in PARAM_FILE_MAPPING.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    self.models[param_key] = joblib.load(model_path)
                except Exception as e:
                    if self.raise_on_missing:
                        raise RuntimeError(f"Failed to load {model_path}: {e}")
                    warnings.warn(
                        f"Failed to load {model_path}: {e}. "
                        f"Will use default for {param_key}.",
                        UserWarning,
                    )
            else:
                if self.raise_on_missing:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def predict(
        self,
        predictions: NDArray[np.integer],
        num_classes: int | None = None,
    ) -> dict[str, Any]:
        """Predict optimal hyperparameters for the given dataset.
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Ensemble predictions where each row is a sample and each column
            is a classifier's prediction.
        num_classes : int, optional
            Number of classes. If None, inferred from data.
            
        Returns
        -------
        dict
            Dictionary with hyperparameter names and predicted values:
            - batch_size: int
            - epochs: int
            - learning_rate: float
            - init_method: str
            - num_layers: int
            - activation_func: str
            - momentum: float
            - scheduler: str
        """
        predictions = np.asarray(predictions)
        
        # Extract meta-features
        if num_classes is not None:
            self.feature_extractor.num_classes = num_classes
        
        features = self.feature_extractor.extract(predictions)
        
        return self.predict_from_features(features)
    
    def predict_from_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict hyperparameters from pre-computed meta-features.
        
        Parameters
        ----------
        features : dict
            Dictionary of meta-features with keys matching FEATURE_NAMES.
            
        Returns
        -------
        dict
            Predicted hyperparameters.
        """
        # Create feature vector in correct order
        try:
            import pandas as pd
            feature_df = pd.DataFrame([features])[FEATURE_NAMES]
        except ImportError:
            # Fallback to numpy if pandas not available
            feature_vector = np.array(
                [[features.get(name, 0.0) for name in FEATURE_NAMES]],
                dtype=np.float64,
            )
            feature_df = feature_vector
        
        # Predict each hyperparameter
        hyperparams = DEFAULT_HYPERPARAMETERS.copy()
        
        for param_key, model in self.models.items():
            try:
                pred = model.predict(feature_df)[0]
                
                # Post-process predictions
                hyperparams[param_key] = self._postprocess_prediction(param_key, pred)
            except Exception as e:
                warnings.warn(
                    f"Failed to predict {param_key}: {e}. Using default.",
                    UserWarning,
                )
        
        return hyperparams
    
    def _postprocess_prediction(self, param_key: str, value: Any) -> Any:
        """Post-process predicted values to ensure correct types and ranges."""
        if param_key == 'batch_size':
            return int(value)
        elif param_key == 'epochs':
            return int(value)
        elif param_key == 'learning_rate':
            return float(value)
        elif param_key == 'num_layers':
            return int(value)
        elif param_key == 'momentum':
            return float(value)
        elif param_key == 'init_method':
            return str(value)
        elif param_key == 'activation_func':
            return str(value)
        elif param_key == 'scheduler':
            # Handle 'nan' or 'clean' values
            val = str(value)
            if val.lower() in ('nan', 'clean'):
                return 'None'
            return val
        
        return value
    
    def predict_as_list(
        self,
        predictions: NDArray[np.integer],
        num_classes: int | None = None,
    ) -> list[Any]:
        """Predict hyperparameters and return as ordered list.
        
        This format matches the order expected by update_config_by_hyperparameters():
        [batch_size, epochs, lr, init_method, num_layers, activation_func, momentum, scheduler]
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Ensemble predictions.
        num_classes : int, optional
            Number of classes.
            
        Returns
        -------
        list
            Hyperparameters in order: [batch_size, epochs, lr, init_method,
            num_layers, activation_func, momentum, scheduler]
        """
        hyperparams = self.predict(predictions, num_classes)
        return [hyperparams[key] for key in HYPERPARAM_ORDER]
    
    def get_available_parameters(self) -> list[str]:
        """Get list of parameters that have loaded models."""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if predictor has models loaded and is ready to predict."""
        return len(self.models) > 0
    
    @staticmethod
    def get_default_hyperparameters() -> dict[str, Any]:
        """Get default hyperparameters used when prediction fails."""
        return DEFAULT_HYPERPARAMETERS.copy()

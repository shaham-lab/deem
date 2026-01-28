"""High-level DEEM API for plug-and-play ensemble aggregation.

This module provides the `DEEM` class - the main user-facing API for
training RBMs on ensemble predictions from multiple classifiers.

Examples
--------
Basic usage with hard labels::

    >>> from deem import DEEM
    >>> import numpy as np
    >>>
    >>> # Ensemble predictions: (n_samples, n_classifiers)
    >>> predictions = np.random.randint(0, 3, (100, 15))
    >>>
    >>> model = DEEM()
    >>> model.fit(predictions)  # Unsupervised training
    >>> consensus = model.predict(predictions)

With labels for evaluation::

    >>> labels = np.random.randint(0, 3, (100,))
    >>> model = DEEM(n_classes=3)
    >>> model.fit(predictions, labels=labels, epochs=50)
    >>> accuracy = model.score(predictions, labels)

With soft labels (probability distributions)::

    >>> # Soft predictions: (n_samples, n_classes, n_classifiers)
    >>> # Each classifier outputs probability distribution over classes
    >>> soft_predictions = np.random.rand(100, 3, 15)
    >>> soft_predictions = soft_predictions / soft_predictions.sum(axis=1, keepdims=True)
    >>>
    >>> model = DEEM(n_classes=3)
    >>> model.fit(soft_predictions)  # Automatically detects 3D and enables oh_mode
    >>> consensus = model.predict(soft_predictions)

Custom hyperparameters::

    >>> model = DEEM(
    ...     n_classes=3,
    ...     hidden_dim=1,
    ...     learning_rate=0.01,
    ...     epochs=100,
    ...     auto_hyperparameters=False  # Use manual params
    ... )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .core.evaluation import (
    accuracy_with_hungarian,
    get_hungarian_solution,
    vectorize_predictions,
)
from .core.losses import EBMLoss
from .core.models import MultinomialRBMGwg
from .core.training import RBMTrainer
from .data import TensorDatasetWithShape, filter_missing_samples

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class DEEM:
    """High-level API for RBM-based ensemble aggregation.

    DEEM (Deep Ensemble Energy Models) provides a scikit-learn style interface
    for training Restricted Boltzmann Machines on ensemble predictions from
    multiple classifiers (crowd learning, model aggregation).

    The model is **UNSUPERVISED** by default - no labels are required for training.
    Labels can optionally be provided for evaluation only.

    Key Features
    ------------
    - **Unsupervised training**: No labels needed, learns from ensemble structure
    - **Automatic buffer initialization**: Initializes sampler with training data
    - **Weighted initialization**: Scales weights by classifier accuracy (default ON)
    - **Soft label support**: Handles probability distributions (3D tensors)
    - **Transparent Hungarian alignment**: Automatic label permutation handling
    - **Auto-hyperparameters**: Optional automatic hyperparameter selection

    Parameters
    ----------
    n_classes : int, optional
        Number of classes. If None, auto-detected from max prediction value
        (for hard labels) or from tensor shape (for soft labels).
    hidden_dim : int, default=1
        Number of hidden units (learned representation size). For standard
        ensemble aggregation, this should be 1 (one consensus prediction).
        **Note**: Research shows hidden_dim=1 gives best results.
    cd_k : int, default=10
        Number of contrastive divergence steps.
    deterministic : bool, default=True
        If True, use probabilities instead of sampling during CD.
        Deterministic mode is more stable for most use cases.
    learning_rate : float, default=0.001
        Learning rate for SGD optimizer.
    momentum : float, default=0.9
        Momentum for SGD optimizer.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=128
        Training batch size.
    device : str, default='auto'
        Device for training: 'auto', 'cpu', or 'cuda'.
        'auto' selects CUDA if available.
    use_weighted : bool, default=True
        If True, scale RBM weights by each classifier's agreement with
        majority vote. This gives better classifiers more influence.
        **Recommended**: Keep True (matches production behavior).
        Set to False only for ablation studies.
    auto_hyperparameters : bool, default=False
        If True, use hyperparameter prediction model to automatically
        select optimal hyperparameters based on dataset meta-features.
        **Works out-of-the-box** using bundled trained models.
    model_dir : str or Path, optional
        Custom directory for hyperparameter prediction models.
        Only needed if you want to use custom trained models.
        Default (None) uses bundled models from the package.
    random_state : int, optional
        Random seed for reproducibility.
    use_preprocessing : bool, default=False
        If True, add learnable Multinomial preprocessing layers before RBM.
        This adds model capacity for complex transformations.
        **Note**: Only enable if auto_hyperparameters predicts num_layers > 0.
    preprocessing_layers : int, default=1
        Number of Multinomial preprocessing layers (only if use_preprocessing=True).
        Ignored if preprocessing_layer_widths is provided.
    preprocessing_activation : str, default='sparsemax'
        Activation function for preprocessing layers. Options:
        'sparsemax', 'entmax', 'ultramax', 'softmax', 'gelu', 'silu'.
    preprocessing_init : str, default='identity'
        Weight initialization for preprocessing layers. Options:
        'identity', 'rand', 'mv' (majority vote), 'default' (Kaiming).
    preprocessing_one_hot : bool, default=False
        If True, output one-hot vectors using hard Gumbel-Softmax.
    preprocessing_use_softmax : bool, default=True
        If True, apply activation function to preprocessing outputs.
    preprocessing_jitter : float, default=0.005
        Jitter coefficient for weight initialization.
    preprocessing_layer_widths : list of int, optional
        Custom layer widths for preprocessing network. If provided,
        creates a network with specified widths (e.g., [20, 15, 10]).
        Overrides preprocessing_layers parameter.
    sampler_steps : int, default=5
        Number of MCMC sampling steps for generating fake samples.
        **Note**: Production default is 5. Higher values = slower but more accurate.
    sampler_oh_mode : bool, default=False
        If True, sampler operates in one-hot mode.
        **Note**: Automatically enabled for soft labels (3D tensors).
    **kwargs
        Additional keyword arguments passed to MultinomialRBMGwg.

    Attributes
    ----------
    model_ : MultinomialRBMGwg or None
        The trained RBM model. None before fit() is called.
    class_map_ : dict or None
        Mapping from predicted to aligned class labels.
        Set after calling predict() with align_to parameter or score().
        **Important**: Alignment is against MAJORITY VOTE, not true labels.
    n_classes_ : int
        Number of classes (inferred or specified).
    n_classifiers_ : int
        Number of classifiers (input dimension).
    history_ : dict
        Training history with loss and learning rate per epoch.
    is_fitted_ : bool
        True if the model has been trained.

    Examples
    --------
    **Basic unsupervised training**::

        >>> from deem import DEEM
        >>> import numpy as np
        >>>
        >>> # Ensemble predictions: (n_samples, n_classifiers)
        >>> predictions = np.random.randint(0, 3, (100, 15))
        >>>
        >>> model = DEEM(n_classes=3)
        >>> model.fit(predictions)  # Unsupervised - no labels needed!
        >>> consensus = model.predict(predictions)

    **With evaluation and alignment**::

        >>> labels = np.random.randint(0, 3, (100,))
        >>>
        >>> model = DEEM(n_classes=3, epochs=50)
        >>> model.fit(predictions)
        >>>
        >>> # Predict with automatic alignment to majority vote
        >>> consensus = model.predict(predictions, align_to=predictions)
        >>>
        >>> # Or use score() which handles alignment automatically
        >>> accuracy = model.score(predictions, labels)
        >>> print(f"Accuracy: {accuracy:.2%}")

    **With soft labels (probability distributions)**::

        >>> # Soft predictions: (n_samples, n_classes, n_classifiers)
        >>> soft_predictions = np.random.rand(100, 3, 15)
        >>> soft_predictions = soft_predictions / soft_predictions.sum(axis=1, keepdims=True)
        >>>
        >>> model = DEEM(n_classes=3)
        >>> model.fit(soft_predictions)  # Automatically detects 3D and enables oh_mode
        >>> consensus = model.predict(soft_predictions)

    **With automatic hyperparameters** (works out-of-the-box!)::

        >>> model = DEEM(
        ...     auto_hyperparameters=True,  # Uses bundled trained models
        ...     epochs=50  # Can still override auto-selected epochs
        ... )
        >>> model.fit(predictions, verbose=True)
        >>> # Hyperparameters automatically selected based on dataset characteristics

    **With preprocessing layers** (for complex datasets)::

        >>> model = DEEM(
        ...     n_classes=3,
        ...     use_preprocessing=True,
        ...     preprocessing_layers=1,
        ...     preprocessing_activation='entmax',
        ...     preprocessing_init='identity'
        ... )
        >>> model.fit(predictions)
        >>> # Model has Multinomial layer(s) before RBM for added capacity

    **Disabling weighted initialization** (for ablation studies)::

        >>> model = DEEM(n_classes=3, use_weighted=False)
        >>> model.fit(predictions)
        >>> # All classifiers treated equally (not recommended for production)

    Notes
    -----
    **Label Permutation Problem**: RBM hidden units have no inherent ordering,
    so predicted class IDs may not match expected class IDs. Use the ``align_to``
    parameter in ``predict()`` or the ``score()`` method to automatically handle
    this via the Hungarian algorithm.

    **Unsupervised Alignment**: Alignment is ALWAYS against majority vote of
    predictions, never against true labels. This maintains the unsupervised
    nature of the model.

    **Buffer Initialization**: The sampler buffer is automatically initialized
    with the first batch of training data to improve MCMC mixing. This happens
    transparently and requires no user configuration.

    **Weighted Initialization**: By default (use_weighted=True), RBM weights
    are scaled by each classifier's accuracy relative to majority vote. This
    gives better classifiers more influence in the ensemble.

    See Also
    --------
    deem.core.models.MultinomialRBMGwg : The underlying RBM model.
    deem.core.training.RBMTrainer : Training loop implementation.
    deem.automl.HyperparameterPredictor : Automatic hyperparameter selection.
    deem.core.evaluation : Hungarian algorithm for label alignment.

    References
    ----------
    .. [1] GitHub repository: https://github.com/shaham-lab/deem
    """

    def __init__(
        self,
        n_classes: Optional[int] = None,
        hidden_dim: int = 1,
        cd_k: int = 10,
        deterministic: bool = True,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        epochs: int = 100,
        batch_size: int = 128,
        device: str = 'auto',
        auto_hyperparameters: bool = False,
        model_dir: Optional[Union[str, Path]] = None,
        random_state: Optional[int] = None,
        # Multinomial preprocessing network parameters
        use_preprocessing: bool = False,
        preprocessing_layers: int = 1,
        preprocessing_activation: str = 'sparsemax',
        preprocessing_init: str = 'identity',
        preprocessing_one_hot: bool = False,
        preprocessing_use_softmax: bool = True,
        preprocessing_jitter: float = 0.005,
        preprocessing_layer_widths: Optional[List[int]] = None,
        # Sampler parameters
        sampler_steps: int = 5,
        sampler_oh_mode: bool = False,
        # Weighted initialization
        use_weighted: bool = True,
        **kwargs,
    ):
        # Store hyperparameters
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.cd_k = cd_k
        self.deterministic = deterministic
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.device_str = device
        self.auto_hyperparameters = auto_hyperparameters
        self.model_dir = model_dir
        self.random_state = random_state
        
        # Multinomial preprocessing network parameters
        self.use_preprocessing = use_preprocessing
        self.preprocessing_layers = preprocessing_layers
        self.preprocessing_activation = preprocessing_activation
        self.preprocessing_init = preprocessing_init
        self.preprocessing_one_hot = preprocessing_one_hot
        self.preprocessing_use_softmax = preprocessing_use_softmax
        self.preprocessing_jitter = preprocessing_jitter
        self.preprocessing_layer_widths = preprocessing_layer_widths
        
        # Sampler parameters
        self.sampler_steps = sampler_steps
        self.sampler_oh_mode = sampler_oh_mode
        
        # Weighted initialization
        self.use_weighted = use_weighted
        
        self.kwargs = kwargs

        # Model state (lazy initialization)
        self.model_: Optional[MultinomialRBMGwg] = None
        self.class_map_: Optional[Dict[int, int]] = None
        self.n_classes_: Optional[int] = None
        self.n_classifiers_: Optional[int] = None
        self.history_: Dict[str, List[float]] = {}
        self.is_fitted_: bool = False
        
        # Soft label state (set during _prepare_data)
        self._is_soft_labels: bool = False

        # Resolve device
        self.device_ = self._resolve_device(device)

        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve 'auto' device to actual device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _to_tensor(
        self, data: ArrayLike, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Convert array-like to tensor on the correct device."""
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self.device_)
        return torch.tensor(data, dtype=dtype, device=self.device_)

    def _prepare_data(
        self,
        predictions: ArrayLike,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Validate and prepare input data.

        Parameters
        ----------
        predictions : array-like
            Input predictions, shape (n_samples, n_classifiers) for hard labels
            or (n_samples, n_classes, n_classifiers) for soft labels.
        labels : array-like, optional
            True labels, shape (n_samples,).

        Returns
        -------
        predictions_tensor : torch.Tensor
            Prepared predictions tensor.
        labels_tensor : torch.Tensor or None
            Prepared labels tensor, or None if not provided.
        """
        # Convert to tensor
        predictions_tensor = self._to_tensor(predictions, torch.float32)
        
        # Handle soft labels (N, K, D) - auto-detect
        if len(predictions_tensor.shape) == 3:
            self._is_soft_labels = True
        else:
            self._is_soft_labels = False

        # Prepare labels if provided
        labels_tensor = None
        if labels is not None:
            labels_tensor = self._to_tensor(labels, torch.long)
            if len(labels_tensor.shape) > 1:
                labels_tensor = labels_tensor.flatten()

        return predictions_tensor, labels_tensor

    def _infer_data_params(self, predictions: torch.Tensor) -> None:
        """Infer n_classes and n_classifiers from data."""
        if self._is_soft_labels:
            # Soft labels: (N, K, D)
            _, self.n_classes_, self.n_classifiers_ = predictions.shape
        else:
            # Hard labels: (N, D)
            _, self.n_classifiers_ = predictions.shape
            if self.n_classes is not None:
                self.n_classes_ = self.n_classes
            else:
                # Infer from max value (excluding -1)
                valid_values = predictions[predictions != -1]
                if len(valid_values) > 0:
                    self.n_classes_ = int(valid_values.max().item()) + 1
                else:
                    raise ValueError("Cannot infer n_classes: all values are missing (-1)")

    def _get_auto_hyperparameters(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Get hyperparameters from the automl predictor."""
        try:
            from .automl import HyperparameterPredictor
            
            # Use model_dir if specified, otherwise use bundled models (None)
            predictor = HyperparameterPredictor(
                model_dir=self.model_dir,  # None = use bundled models
                raise_on_missing=False
            )
            
            if predictor.is_ready():
                hyperparams = predictor.predict(predictions, self.n_classes_)
                return hyperparams
            else:
                warnings.warn(
                    "Hyperparameter prediction models not loaded. "
                    "Using default hyperparameters.",
                    UserWarning
                )
                return {}
        except Exception as e:
            warnings.warn(
                f"Failed to predict hyperparameters: {e}. Using defaults.",
                UserWarning
            )
            return {}

    def _init_model(self) -> None:
        """Initialize the RBM model with current parameters."""
        if self.n_classes_ is None or self.n_classifiers_ is None:
            raise RuntimeError("Must call _infer_data_params() before _init_model()")

        # Build multinomial preprocessing network if requested
        multinomial_net = None
        rbm_dx = self.n_classifiers_
        
        if self.use_preprocessing:
            from deem.core.preprocessing import Multinomial
            import torch.nn as nn
            
            layers = []
            
            # Option 1: Use layer_widths if provided
            if self.preprocessing_layer_widths is not None and len(self.preprocessing_layer_widths) > 0:
                input_dim = self.n_classifiers_
                for width in self.preprocessing_layer_widths:
                    layers.append(
                        Multinomial(
                            in_multi_units=self.n_classes_,
                            out_multi_units=self.n_classes_,
                            in_features=input_dim,
                            out_features=width,
                            init_method=self.preprocessing_init,
                            one_hot=self.preprocessing_one_hot,
                            use_softmax=self.preprocessing_use_softmax,
                            device=self.device_,
                            jitter_coef=self.preprocessing_jitter,
                            activation_func_name=self.preprocessing_activation,
                        )
                    )
                    input_dim = width
                rbm_dx = self.preprocessing_layer_widths[-1]
            
            # Option 2: Use num_layers with fixed width (original behavior)
            elif self.preprocessing_layers > 0:
                for _ in range(self.preprocessing_layers):
                    layers.append(
                        Multinomial(
                            in_multi_units=self.n_classes_,
                            out_multi_units=self.n_classes_,
                            in_features=self.n_classifiers_,
                            out_features=self.n_classifiers_,
                            init_method=self.preprocessing_init,
                            one_hot=self.preprocessing_one_hot,
                            use_softmax=self.preprocessing_use_softmax,
                            device=self.device_,
                            jitter_coef=self.preprocessing_jitter,
                            activation_func_name=self.preprocessing_activation,
                        )
                    )
            
            if layers:
                multinomial_net = nn.Sequential(*layers)

        # Extract init_method from kwargs to avoid duplicate argument error
        init_method = self.kwargs.pop('init_method', 'mv_rand')
        
        self.model_ = MultinomialRBMGwg(
            dx=rbm_dx,
            dh=self.hidden_dim,
            k=self.n_classes_,
            l=self.n_classes_,
            m=self.n_classes_,
            device=self.device_,
            cd_k=self.cd_k,
            deterministic=self.deterministic,
            multinomial_net=multinomial_net,
            sampler_steps=self.sampler_steps,
            oh_mode=self.sampler_oh_mode,
            init_method=init_method,  # Use extracted value with fallback to mv_rand
            **self.kwargs
        )
        self.model_.to(self.device_)

    def _initialize_buffer_with_first_batch(
        self,
        train_loader: DataLoader,
        verbose: bool = False,
    ) -> None:
        """Initialize sampler buffer with first batch of training data.

        This improves MCMC mixing by starting with real examples instead
        of random noise. Matches production behavior in run_predict.py
        (lines 371-377).

        The buffer is initialized with:
        - Raw data if oh_mode is False
        - Preprocessed (one-hot) data if oh_mode is True

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        verbose : bool, default=False
            If True, print buffer initialization info.

        Notes
        -----
        This method is a no-op if the buffer is already populated.
        """
        # Check if buffer already populated (prevent duplicate initialization)
        if len(self.model_.sampler.buffer) > 0:
            if verbose:
                print(f"  (Buffer already initialized with {len(self.model_.sampler.buffer)} examples, skipping)")
            return

        # Create fixed loader (no shuffle) to get consistent first batch
        fixed_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
        )

        # Extract first batch
        first_batch = next(iter(fixed_loader))

        # Handle batch format (may be tuple or tensor)
        if isinstance(first_batch, (list, tuple)):
            batch_data = first_batch[0]
        else:
            batch_data = first_batch

        # Move to correct device
        batch_data = batch_data.to(self.device_)

        # Initialize buffer based on oh_mode
        # oh_mode=True means input is already one-hot encoded (soft labels)
        # oh_mode=False means input needs preprocessing
        if not self.model_.sampler.oh_mode:
            # Standard mode: add raw batch (will be preprocessed during sampling)
            self.model_.sampler.buffer.add_examples(batch_data.cpu())
        else:
            # One-hot mode: preprocess batch first before adding
            preprocessed = self.model_.preprocess(batch_data)
            self.model_.sampler.buffer.add_examples(preprocessed.cpu())

        if verbose:
            buffer_size = len(self.model_.sampler.buffer)
            print(f"Initialized sampler buffer with {buffer_size} examples")

    def _compute_majority_vote(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Compute majority vote for each sample.

        Parameters
        ----------
        predictions : torch.Tensor
            Predictions tensor, shape (n_samples, n_classifiers) for hard labels
            or (n_samples, n_classes, n_classifiers) for soft labels.

        Returns
        -------
        majority_vote : torch.Tensor
            Majority vote predictions, shape (n_samples,).
        """
        # Handle soft labels (3D)
        if len(predictions.shape) == 3:
            # Convert to hard labels via argmax
            predictions = predictions.argmax(dim=1)  # (N, D)

        # Compute mode (most frequent value) for each sample
        majority_vote = torch.mode(predictions.long(), dim=1).values

        return majority_vote

    def _apply_weighted_initialization(
        self,
        train_loader: DataLoader,
        verbose: bool = False,
    ) -> None:
        """Scale RBM weights by classifier accuracy relative to majority vote.

        This gives better classifiers more influence in the energy function,
        improving training quality. Matches production behavior in run_predict.py
        (lines 355-360).

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        verbose : bool, default=False
            If True, print per-classifier accuracies.

        Notes
        -----
        This method is a no-op if use_weighted=False.

        The weighting is computed as:
        1. Compute majority vote for all training examples
        2. For each classifier, compute accuracy vs majority vote
        3. Scale RBM weights: weights *= accuracy_vector.view(1, 1, -1, 1)

        This gives classifiers that agree with the majority higher weight
        in the energy function.
        """
        if not self.use_weighted:
            return

        # Create fixed loader (no shuffle) to get all data
        fixed_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
        )

        # Collect all training data
        all_batches = []
        for batch in fixed_loader:
            if isinstance(batch, (list, tuple)):
                all_batches.append(batch[0])
            else:
                all_batches.append(batch)

        all_data = torch.cat(all_batches, dim=0).to(self.device_)

        # Handle soft labels (3D tensors)
        if self._is_soft_labels:
            # Soft labels: (N, K, D) -> convert to hard labels via argmax
            all_data = all_data.argmax(dim=1)  # -> (N, D)

        # Compute majority vote for each sample
        # torch.mode returns (values, indices), we want values
        mv_preds = torch.mode(all_data, dim=1).values

        # Compute per-classifier accuracy vs majority vote
        # all_data: (N, D), mv_preds: (N,)
        # Need to broadcast: mv_preds.unsqueeze(1) -> (N, 1)
        matches = (all_data == mv_preds.unsqueeze(1)).float()
        clf_accuracy_vector = matches.mean(dim=0)  # Average over samples -> (D,)

        # Scale RBM weights by accuracy vector
        # model.weights has shape (dh, m, dx, k)
        # Need to broadcast accuracy to match weight dimensions
        # clf_accuracy_vector: (D,) -> (1, 1, D, 1)
        with torch.no_grad():
            weight_scale = clf_accuracy_vector.view(1, 1, -1, 1)
            self.model_.weights *= weight_scale

        if verbose:
            print(f"\nWeighted initialization applied:")
            # Print accuracies as comma-separated one-liner
            acc_str = ", ".join([f"{i}:{acc.item():.3f}" for i, acc in enumerate(clf_accuracy_vector)])
            print(f"  Classifier accuracies: {acc_str}")
            print(f"  Mean: {clf_accuracy_vector.mean().item():.3f}, Std: {clf_accuracy_vector.std().item():.3f}")

            # Warn about low-performing classifiers
            low_performers = (clf_accuracy_vector < 0.1).sum().item()
            if low_performers > 0:
                print(f"  ⚠️  Warning: {low_performers} classifier(s) with <10% accuracy")

    def fit(
        self,
        predictions: ArrayLike,
        labels: Optional[ArrayLike] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        val_predictions: Optional[ArrayLike] = None,
        val_labels: Optional[ArrayLike] = None,
    ) -> 'DEEM':
        """Train the RBM on ensemble predictions.

        The model is trained in an UNSUPERVISED manner - labels are optional
        and only used for monitoring/early stopping.

        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Predictions from multiple classifiers. Values should be integers
            0 to k-1, or -1 for missing predictions.
            Can also be soft labels of shape (n_samples, n_classes, n_classifiers).
        labels : array-like, optional, shape (n_samples,)
            True labels. Optional - only used for monitoring.
        epochs : int, optional
            Override the number of epochs. If None, uses self.epochs.
        learning_rate : float, optional
            Override learning rate. If None, uses self.learning_rate.
        batch_size : int, optional
            Override batch size. If None, uses self.batch_size.
        verbose : bool, default=True
            Print training progress.
        val_predictions : array-like, optional
            Validation predictions for monitoring.
        val_labels : array-like, optional
            Validation labels for monitoring.

        Returns
        -------
        self : DEEM
            The fitted model (for method chaining).

        Examples
        --------
        >>> model = DEEM()
        >>> model.fit(predictions)  # Unsupervised

        >>> model = DEEM()
        >>> model.fit(predictions, labels=labels, epochs=50)  # With labels
        """
        # Prepare data
        predictions_tensor, labels_tensor = self._prepare_data(predictions, labels)

        # Filter missing samples (all -1) - only if using hard labels
        predictions_np = predictions_tensor.cpu().numpy()
        
        if not self._is_soft_labels:
            # Only filter for hard labels (soft labels don't have -1)
            if labels_tensor is not None:
                labels_np = labels_tensor.cpu().numpy()
                predictions_filtered, labels_filtered = filter_missing_samples(
                    predictions_np, labels_np, missing_value=-1
                )
                predictions_tensor = self._to_tensor(predictions_filtered)
                labels_tensor = self._to_tensor(labels_filtered, torch.long)
            else:
                # No labels - filter predictions only
                valid_mask = np.any(predictions_np != -1, axis=1)
                predictions_filtered = predictions_np[valid_mask]
                predictions_tensor = self._to_tensor(predictions_filtered)

        # Infer data parameters
        self._infer_data_params(predictions_tensor)

        # Auto hyperparameters if enabled
        _epochs = epochs if epochs is not None else self.epochs
        _lr = learning_rate if learning_rate is not None else self.learning_rate
        _batch_size = batch_size if batch_size is not None else self.batch_size

        if self.auto_hyperparameters:
            auto_params = self._get_auto_hyperparameters(predictions_np)
            if auto_params:
                # Apply training hyperparameters
                _epochs = epochs if epochs is not None else auto_params.get('epochs', _epochs)
                _lr = learning_rate if learning_rate is not None else auto_params.get('learning_rate', _lr)
                _batch_size = batch_size if batch_size is not None else auto_params.get('batch_size', _batch_size)
                
                # Apply model architecture hyperparameters (Multinomial network)
                if 'num_layers' in auto_params and auto_params['num_layers'] > 0:
                    self.use_preprocessing = True
                    self.preprocessing_layers = auto_params['num_layers']
                if 'activation_func' in auto_params:
                    self.preprocessing_activation = auto_params['activation_func']
                if 'init_method' in auto_params:
                    # Store for kwargs - this is RBM init_method
                    if 'init_method' not in self.kwargs:
                        self.kwargs['init_method'] = auto_params['init_method']
                if 'momentum' in auto_params:
                    self.momentum = auto_params['momentum']
                
                if verbose:
                    print(f"Auto-selected hyperparameters:")
                    print(f"  Training: epochs={_epochs}, lr={_lr}, batch_size={_batch_size}, momentum={self.momentum}")
                    if self.use_preprocessing:
                        print(f"  Architecture: num_layers={self.preprocessing_layers}, activation={self.preprocessing_activation}")
                    if 'init_method' in auto_params:
                        print(f"  Initialization: {auto_params['init_method']}")
                    if 'scheduler' in auto_params and auto_params['scheduler'] not in ('None', 'nan', 'clean'):
                        print(f"  Scheduler: {auto_params['scheduler']}")

        # Auto-enable oh_mode for soft labels (3D tensors)
        # This matches production behavior in run_predict.py (lines 363-370)
        if self._is_soft_labels:
            self.sampler_oh_mode = True
            if verbose:
                print("Soft labels (3D) detected - enabled one-hot sampler mode")

        # Initialize model
        self._init_model()

        # Create DataLoader
        if labels_tensor is not None:
            dataset = TensorDatasetWithShape(predictions_tensor, labels_tensor)
        else:
            dataset = TensorDataset(predictions_tensor)

        train_loader = DataLoader(
            dataset,
            batch_size=_batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Initialize buffer with first batch for better MCMC mixing
        # This matches production behavior in run_predict.py
        self._initialize_buffer_with_first_batch(train_loader, verbose=verbose)

        # Apply weighted initialization (scale weights by classifier accuracy)
        # This matches production behavior in run_predict.py (use_weighted=1)
        self._apply_weighted_initialization(train_loader, verbose=verbose)

        # Create optimizer and loss
        optimizer = torch.optim.SGD(
            self.model_.parameters(),
            lr=_lr,
            momentum=self.momentum,
        )
        loss_fn = EBMLoss(self.model_)

        # Create trainer and train
        trainer = RBMTrainer(
            model=self.model_,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device_,
        )

        # Store training data for accuracy monitoring if verbose
        # Mark as fitted so predict() works during training
        self.is_fitted_ = True
        
        if verbose and labels_tensor is not None:
            # Compute initial accuracy before training
            initial_preds = self.predict(predictions_tensor.cpu().numpy())
            initial_acc = (initial_preds == labels_tensor.cpu().numpy()).mean()
            print(f"\nInitial accuracy (before training): {initial_acc:.2%}")
            
            # Store for epoch-wise monitoring
            self._training_data = (predictions_tensor.cpu().numpy(), labels_tensor.cpu().numpy())
        else:
            self._training_data = None

        self.history_ = trainer.fit(
            train_loader=train_loader,
            epochs=_epochs,
            verbose=verbose,
            accuracy_callback=self._compute_epoch_accuracy if verbose else None,
        )

        # Clean up
        self._training_data = None

        return self

    def predict(
        self,
        predictions: ArrayLike,
        return_probs: bool = False,
        align_to: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Predict consensus labels with automatic Hungarian alignment.

        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Predictions from multiple classifiers.
        return_probs : bool, default=False
            If True, return class probabilities instead of hard labels.
        align_to : array-like, optional
            Reference predictions for alignment. If None, uses predictions itself.
            The Hungarian algorithm aligns against majority vote of this reference.

        Returns
        -------
        consensus : np.ndarray
            Consensus predictions of shape (n_samples,) or
            probabilities of shape (n_samples, n_classes) if return_probs=True.
            Predictions are automatically aligned to majority vote unless
            return_probs=True.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        Notes
        -----
        **Automatic Alignment**: Hungarian alignment is ALWAYS computed against
        majority vote (not true labels). This is an inherent part of the algorithm.
        
        **Label Permutation Problem**: RBM hidden units have no inherent ordering,
        so predicted class IDs may not match your expected class IDs. The Hungarian
        algorithm solves this by finding the optimal label mapping.

        Examples
        --------
        Basic prediction (automatically aligned to MV)::

            >>> consensus = model.predict(test_predictions)

        Prediction aligned to training data MV::

            >>> # Align to majority vote of training data
            >>> consensus = model.predict(test_predictions, align_to=train_predictions)
        """
        self._check_is_fitted()

        predictions_tensor, _ = self._prepare_data(predictions)

        self.model_.eval()
        with torch.no_grad():
            # Preprocess and get hidden probabilities
            x = self.model_.preprocess(predictions_tensor)
            x = self.model_.apply_multinomial_layer(x)
            h_probs = self.model_.calc_hidden_probs(x)

            if return_probs:
                # Squeeze hidden dimension if it's 1
                # Note: No alignment for probabilities
                probs = h_probs.squeeze(1).cpu().numpy()
                return probs
            else:
                # Argmax over class dimension to get predictions
                # h_probs has shape (N, dh, m) - squeeze dh if 1
                raw_predictions = h_probs.argmax(dim=-1).squeeze(-1).cpu().numpy()

                # ALWAYS compute Hungarian alignment against majority vote
                # This is an inherent part of the algorithm
                align_data = align_to if align_to is not None else predictions
                align_to_tensor, _ = self._prepare_data(align_data)
                reference_mv = self._compute_majority_vote(align_to_tensor)
                reference_mv_np = reference_mv.cpu().numpy()

                # Get raw predictions for reference data for alignment
                x_ref = self.model_.preprocess(align_to_tensor)
                x_ref = self.model_.apply_multinomial_layer(x_ref)
                h_probs_ref = self.model_.calc_hidden_probs(x_ref)
                raw_ref_predictions = h_probs_ref.argmax(dim=-1).squeeze(-1).cpu().numpy()

                # Get Hungarian mapping
                self.class_map_ = get_hungarian_solution(
                    raw_ref_predictions, reference_mv_np, self.n_classes_
                )

                # Apply mapping to requested predictions
                aligned_predictions = vectorize_predictions(raw_predictions, self.class_map_)
                return aligned_predictions

    def get_class_mapping(self) -> Optional[Dict[int, int]]:
        """Get the cached Hungarian class mapping.

        Returns
        -------
        class_map : dict or None
            Mapping from predicted class IDs to aligned class IDs.
            None if no alignment has been computed.

        Examples
        --------
        >>> model.predict(test_preds, align_to=train_preds)
        >>> mapping = model.get_class_mapping()
        >>> print(mapping)  # {0: 2, 1: 0, 2: 1, ...}
        """
        return self.class_map_

    def reset_class_mapping(self) -> None:
        """Clear the cached Hungarian class mapping.

        Use this to force recomputation of alignment on next predict() call.

        Examples
        --------
        >>> model.reset_class_mapping()
        >>> # Next predict() with align_to will compute fresh alignment
        """
        self.class_map_ = None

    def score(
        self,
        predictions: ArrayLike,
        true_labels: ArrayLike,
    ) -> float:
        """Compute accuracy with automatic Hungarian alignment.

        This method automatically handles the label permutation problem
        by aligning predictions against the majority vote of the input
        predictions (NOT against true_labels, maintaining unsupervised nature).

        Parameters
        ----------
        predictions : array-like, shape (n_samples, n_classifiers)
            Predictions from multiple classifiers.
        true_labels : array-like, shape (n_samples,)
            True labels (used only for accuracy computation, not alignment).

        Returns
        -------
        accuracy : float
            Classification accuracy (0.0 to 1.0).

        Notes
        -----
        Alignment is against MAJORITY VOTE of predictions, not true_labels.
        This preserves the unsupervised nature of the model.

        Examples
        --------
        >>> accuracy = model.score(test_predictions, test_labels)
        >>> print(f"Accuracy: {accuracy:.2%}")
        """
        self._check_is_fitted()

        # Convert labels to numpy
        if isinstance(true_labels, torch.Tensor):
            true_labels_np = true_labels.cpu().numpy()
        else:
            true_labels_np = np.asarray(true_labels)

        # Get aligned predictions (align to majority vote of predictions)
        aligned_predictions = self.predict(predictions, align_to=predictions)

        # Compute accuracy
        accuracy = float((aligned_predictions == true_labels_np).mean())

        return accuracy

    def _compute_epoch_accuracy(self) -> Optional[float]:
        """Compute current accuracy on training data (for verbose monitoring)."""
        if self._training_data is None:
            return None
        
        predictions, labels = self._training_data
        try:
            preds = self.predict(predictions)
            accuracy = (preds == labels).mean()
            return accuracy
        except:
            return None

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict() or score()."
            )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_classes': self.n_classes,
            'hidden_dim': self.hidden_dim,
            'cd_k': self.cd_k,
            'deterministic': self.deterministic,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': self.device_str,
            'auto_hyperparameters': self.auto_hyperparameters,
            'model_dir': self.model_dir,
            'random_state': self.random_state,
        }

    def set_params(self, **params) -> 'DEEM':
        """Set parameters for this estimator (sklearn compatibility).

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DEEM
            Estimator instance.
        """
        # Map 'device' to internal attribute 'device_str'
        param_mapping = {'device': 'device_str'}
        
        for key, value in params.items():
            attr_name = param_mapping.get(key, key)
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)
            else:
                raise ValueError(f"Invalid parameter '{key}'")
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : str or Path
            Path to save the model.

        Examples
        --------
        >>> model.save('my_model.pt')
        """
        self._check_is_fitted()

        checkpoint = {
            'model_state_dict': self.model_.state_dict(),
            'n_classes_': self.n_classes_,
            'n_classifiers_': self.n_classifiers_,
            'class_map_': self.class_map_,
            'history_': self.history_,
            'params': self.get_params(),
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> 'DEEM':
        """Load a model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved model.

        Returns
        -------
        self : DEEM
            The loaded model.

        Examples
        --------
        >>> model = DEEM()
        >>> model.load('my_model.pt')
        """
        checkpoint = torch.load(path, map_location=self.device_)

        # Restore parameters
        params = checkpoint.get('params', {})
        self.set_params(**params)

        # Restore state
        self.n_classes_ = checkpoint['n_classes_']
        self.n_classifiers_ = checkpoint['n_classifiers_']
        self.class_map_ = checkpoint.get('class_map_')
        self.history_ = checkpoint.get('history_', {})

        # Re-initialize and load model
        self._init_model()
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted_ = True

        return self

    def __repr__(self) -> str:
        """Return string representation."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"DEEM("
            f"n_classes={self.n_classes}, "
            f"hidden_dim={self.hidden_dim}, "
            f"epochs={self.epochs}, "
            f"device='{self.device_str}'"
            f") [{status}]"
        )

"""Data loading utilities for ensemble predictions."""

import scipy.io
import torch
import numpy as np
from torch.utils.data import DataLoader
from .datasets import TensorDatasetWithShape
from .transforms import filter_missing_samples


def load_from_mat(
    filepath: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    test_split: float = 0.2,
    shuffle: bool = True,
    combined_dataset: bool = False
):
    """
    Load hard labels from .mat file.
    
    Expected format:
    - 'f': (N, D) or (D, N) array with classifier predictions (integers 0 to k-1, or -1 for missing)
    - 'y': (N,) array with true labels
    - 'k': (optional) number of classes
    
    Note: The data is transposed if needed (legacy format stores as D x N).
    
    Parameters
    ----------
    filepath : str
        Path to .mat file
    batch_size : int
        Batch size for DataLoaders
    val_split : float
        Fraction of training data for validation
    test_split : float
        Fraction of data for testing (if no X_test in file)
    shuffle : bool
        Whether to shuffle training data
    combined_dataset : bool
        If True, use same data for train/val/test
        
    Returns
    -------
    train_loader, val_loader, test_loader : tuple of DataLoader
        DataLoaders for training, validation, and testing
    metadata : dict
        Dataset metadata (n_samples, n_classifiers, n_classes, etc.)
    """
    # Load .mat file
    data = scipy.io.loadmat(filepath)
    
    predictions = data['f']
    labels = data['y'].flatten()
    
    # Handle transposed data (legacy format: D x N instead of N x D)
    # Heuristic: if first dim is smaller, it's probably transposed
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    
    # Filter samples with missing labels
    existing_labels_mask = labels != -1
    predictions = predictions[existing_labels_mask]
    labels = labels[existing_labels_mask]
    
    # Filter samples with all missing values
    predictions, labels = filter_missing_samples(predictions, labels, missing_value=-1)
    
    # Infer number of classes
    if 'k' in data:
        k_val = data['k']
        # Handle both scalar and array formats
        n_classes = int(k_val.item() if hasattr(k_val, 'item') else k_val.flat[0])
    else:
        n_classes = int(predictions[predictions != -1].max() + 1)
    
    n_samples, n_classifiers = predictions.shape
    
    # Convert to tensors
    predictions_tensor = torch.from_numpy(predictions).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create full dataset
    full_dataset = TensorDatasetWithShape(predictions_tensor, labels_tensor)
    
    if combined_dataset:
        # Use same data for all splits
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Check for separate test set in file
        if 'X_test' in data:
            # Separate test set provided
            test_predictions = data['X_test']
            test_labels = data['y_test'].flatten()
            
            # Handle transposed test data
            if test_predictions.shape[0] < test_predictions.shape[1]:
                test_predictions = test_predictions.T
                
            test_predictions, test_labels = filter_missing_samples(
                test_predictions, test_labels, missing_value=-1
            )
            test_dataset = TensorDatasetWithShape(
                torch.from_numpy(test_predictions).float(),
                torch.from_numpy(test_labels).long()
            )
            train_val_dataset = full_dataset
            train_val_size = len(full_dataset)
        else:
            # Split from full data
            test_size = int(len(full_dataset) * test_split)
            train_val_size = len(full_dataset) - test_size
            train_val_dataset = torch.utils.data.Subset(full_dataset, range(train_val_size))
            test_dataset = torch.utils.data.Subset(full_dataset, range(train_val_size, len(full_dataset)))
        
        # Split train into train/val
        val_size = int(train_val_size * val_split)
        train_size = train_val_size - val_size
        
        if isinstance(train_val_dataset, torch.utils.data.Subset):
            train_dataset = torch.utils.data.Subset(
                full_dataset, list(range(train_size))
            )
            val_dataset = torch.utils.data.Subset(
                full_dataset, list(range(train_size, train_val_size))
            )
        else:
            train_dataset = torch.utils.data.Subset(train_val_dataset, list(range(train_size)))
            val_dataset = torch.utils.data.Subset(train_val_dataset, list(range(train_size, train_val_size)))
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Metadata
    metadata = {
        'n_samples': n_samples,
        'n_classifiers': n_classifiers,
        'n_classes': n_classes,
        'format': 'hard_labels',
        'shape': (n_samples, n_classifiers)
    }
    
    return train_loader, val_loader, test_loader, metadata


def load_from_mat_soft(
    filepath: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    test_split: float = 0.2,
    shuffle: bool = True
):
    """
    Load soft labels (probability distributions) from .mat file.
    
    Expected format:
    - 'f': (N, K, D) array with class probabilities
    - 'y': (N,) array with true labels
    - 'k': number of classes
    
    Parameters
    ----------
    filepath : str
        Path to .mat file
    batch_size : int
        Batch size for DataLoaders
    val_split : float
        Fraction of training data for validation
    test_split : float
        Fraction of data for testing
    shuffle : bool
        Shuffle training data
        
    Returns
    -------
    train_loader, val_loader, test_loader : tuple of DataLoader
    metadata : dict
        Dataset information including shape=(N, K, D)
    """
    # Load .mat file
    data = scipy.io.loadmat(filepath)
    
    predictions = data['f']  # (N, K, D)
    if len(predictions.shape) != 3:
        raise ValueError(f"Expected 3D soft labels (N, K, D), got shape {predictions.shape}")
    
    labels = data['y'].flatten()
    
    # Filter samples with missing labels
    existing_labels_mask = labels != -1
    predictions = predictions[existing_labels_mask]
    labels = labels[existing_labels_mask]
    
    # Filter samples with all missing values (rare for soft labels)
    predictions, labels = filter_missing_samples(predictions, labels, missing_value=-1)
    
    n_samples, n_classes, n_classifiers = predictions.shape
    
    # Override k from data if provided
    if 'k' in data:
        k_val = data['k']
        # Handle both scalar and array formats
        n_classes = int(k_val.item() if hasattr(k_val, 'item') else k_val.flat[0])
    
    # Convert to tensors
    predictions_tensor = torch.from_numpy(predictions).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # Create dataset
    full_dataset = TensorDatasetWithShape(predictions_tensor, labels_tensor)
    
    # Split into train/val/test
    test_size = int(len(full_dataset) * test_split)
    train_val_size = len(full_dataset) - test_size
    
    val_size = int(train_val_size * val_split)
    train_size = train_val_size - val_size
    
    train_dataset = torch.utils.data.Subset(full_dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(full_dataset, list(range(train_size, train_val_size)))
    test_dataset = torch.utils.data.Subset(full_dataset, list(range(train_val_size, len(full_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    metadata = {
        'n_samples': n_samples,
        'n_classifiers': n_classifiers,
        'n_classes': n_classes,
        'format': 'soft_labels',
        'shape': (n_samples, n_classes, n_classifiers)
    }
    
    return train_loader, val_loader, test_loader, metadata


def load_dataset(
    data,
    labels=None,
    soft_labels=None,
    batch_size: int = 128,
    val_split: float = 0.1,
    test_split: float = 0.2,
    shuffle: bool = True,
    combined_dataset: bool = False
):
    """
    Unified data loading API with automatic format detection.
    
    This is the recommended entry point for loading ensemble prediction data.
    It automatically detects whether data contains hard labels (N, D) or
    soft labels (N, K, D) based on the tensor shape.
    
    Parameters
    ----------
    data : str, torch.Tensor, or np.ndarray
        - If str: path to .mat file
        - If tensor/array: predictions of shape (N, D) or (N, K, D)
    labels : torch.Tensor or np.ndarray, optional
        True labels of shape (N,). Required if data is tensor/array.
    soft_labels : bool or None
        - None (default): Auto-detect from data shape
        - True: Force soft labels mode (N, K, D)
        - False: Force hard labels mode (N, D)
    batch_size : int
        Batch size for DataLoaders
    val_split : float
        Fraction of training data for validation
    test_split : float
        Fraction of data for testing
    shuffle : bool
        Shuffle training data
    combined_dataset : bool
        If True, use same data for train/val/test (for debugging)
        
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch DataLoaders for training, validation, and testing
    metadata : dict
        Dataset information including:
        - n_samples: number of samples
        - n_classifiers: number of classifiers (D)
        - n_classes: number of classes (K)
        - format: 'hard_labels' or 'soft_labels'
        - shape: original data shape
        
    Examples
    --------
    >>> # From .mat file (auto-detect)
    >>> train, val, test, meta = load_dataset('data.mat')
    
    >>> # From tensors (hard labels)
    >>> predictions = torch.randint(0, 3, (1000, 15))
    >>> labels = torch.randint(0, 3, (1000,))
    >>> train, val, test, meta = load_dataset(predictions, labels)
    
    >>> # From tensors (soft labels)
    >>> soft_preds = torch.rand(1000, 3, 15)  # (N, K, D)
    >>> train, val, test, meta = load_dataset(soft_preds, labels)
    
    >>> # Force specific mode
    >>> train, val, test, meta = load_dataset('data.mat', soft_labels=True)
    """
    # Case 1: .mat file path
    if isinstance(data, str):
        # Detect soft vs hard from file
        mat_data = scipy.io.loadmat(data)
        predictions_shape = mat_data['f'].shape
        
        if soft_labels is None:
            # Auto-detect: 3D = soft, 2D = hard
            soft_labels = (len(predictions_shape) == 3)
        
        if soft_labels:
            return load_from_mat_soft(data, batch_size, val_split, test_split, shuffle)
        else:
            return load_from_mat(data, batch_size, val_split, test_split, shuffle, combined_dataset)
    
    # Case 2: Tensor or array
    if labels is None:
        raise ValueError("labels must be provided when data is a tensor/array")
    
    # Convert to tensors
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(np.asarray(data)).float()
    else:
        data = data.float()
        
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.asarray(labels)).long()
    else:
        labels = labels.long()
    
    # Flatten labels if needed
    if len(labels.shape) > 1:
        labels = labels.flatten()
    
    # Auto-detect soft labels
    if soft_labels is None:
        soft_labels = (len(data.shape) == 3)
    
    # Filter samples with all missing values
    data, labels = filter_missing_samples(data, labels)
    
    # Ensure tensors after filtering
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).long()
    
    # Create dataset
    full_dataset = TensorDatasetWithShape(data, labels)
    
    if combined_dataset:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Split into train/val/test
        test_size = int(len(full_dataset) * test_split)
        train_val_size = len(full_dataset) - test_size
        val_size = int(train_val_size * val_split)
        train_size = train_val_size - val_size
        
        indices = list(range(len(full_dataset)))
        train_dataset = torch.utils.data.Subset(full_dataset, indices[:train_size])
        val_dataset = torch.utils.data.Subset(full_dataset, indices[train_size:train_val_size])
        test_dataset = torch.utils.data.Subset(full_dataset, indices[train_val_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute metadata
    if soft_labels:
        n_samples, n_classes, n_classifiers = data.shape
        format_str = 'soft_labels'
    else:
        n_samples, n_classifiers = data.shape
        # Infer n_classes from data (max value + 1, ignoring -1)
        valid_values = data[data != -1]
        if len(valid_values) > 0:
            n_classes = int(valid_values.max().item() + 1)
        else:
            n_classes = 0
        format_str = 'hard_labels'
    
    metadata = {
        'n_samples': n_samples,
        'n_classifiers': n_classifiers,
        'n_classes': n_classes,
        'format': format_str,
        'shape': tuple(data.shape)
    }
    
    return train_loader, val_loader, test_loader, metadata

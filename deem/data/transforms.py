"""Data transformations for ensemble predictions."""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from .datasets import TensorDatasetWithShape


def convert_soft_to_hard(dataloader: DataLoader) -> DataLoader:
    """
    Convert DataLoader with soft labels (N, K, D) to hard labels (N, D).
    
    Takes argmax over the K (class) dimension to convert probability
    distributions into discrete class predictions.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader with shape (N, K, D) predictions
        
    Returns
    -------
    DataLoader
        New DataLoader with shape (N, D) hard predictions
        
    Raises
    ------
    ValueError
        If input is not 3D (soft labels)
        
    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from deem.data import TensorDatasetWithShape, convert_soft_to_hard
    >>> soft_data = torch.rand(50, 3, 10)  # (N, K, D)
    >>> labels = torch.randint(0, 3, (50,))
    >>> dataset = TensorDatasetWithShape(soft_data, labels)
    >>> soft_loader = DataLoader(dataset, batch_size=10)
    >>> hard_loader = convert_soft_to_hard(soft_loader)
    >>> batch, _ = next(iter(hard_loader))
    >>> batch.shape
    torch.Size([10, 10])
    """
    dataset = dataloader.dataset
    
    # Extract tensors (handle Subset wrapper)
    if isinstance(dataset, Subset):
        full_X = dataset.dataset.tensors[0][dataset.indices]
        full_Y = dataset.dataset.tensors[1][dataset.indices]
    else:
        full_X = dataset.tensors[0]
        full_Y = dataset.tensors[1]
    
    # Validate input shape
    if len(full_X.shape) != 3:
        raise ValueError(f"Expected 3D input (N, K, D), got shape {full_X.shape}")
    
    # Argmax over K dimension: (N, K, D) → (N, D)
    X_hard = torch.argmax(full_X, dim=1).float()
    
    # Create new dataset
    new_dataset = TensorDatasetWithShape(X_hard, full_Y)
    
    # Preserve shuffle status
    is_shuffled = isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
    
    # Create new loader
    return DataLoader(
        new_dataset,
        batch_size=dataloader.batch_size,
        shuffle=is_shuffled,
        num_workers=dataloader.num_workers
    )


def filter_missing_samples(data, labels, missing_value: int = -1):
    """
    Remove samples where ALL classifiers failed (all values = missing_value).
    
    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Predictions of shape (N, D) or (N, K, D)
    labels : torch.Tensor or np.ndarray
        Labels of shape (N,)
    missing_value : int, default=-1
        Value indicating missing prediction
        
    Returns
    -------
    filtered_data, filtered_labels : tuple
        Data and labels with all-missing samples removed
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([
    ...     [0, 1, 2],  # Valid
    ...     [-1, -1, -1],  # All missing
    ...     [1, 0, 1],  # Valid
    ... ])
    >>> labels = np.array([0, 1, 2])
    >>> filtered_data, filtered_labels = filter_missing_samples(data, labels)
    >>> len(filtered_data)
    2
    """
    # Track input types
    data_is_tensor = isinstance(data, torch.Tensor)
    labels_is_tensor = isinstance(labels, torch.Tensor)
    
    # Convert to numpy for easier handling
    if data_is_tensor:
        data_np = data.numpy()
    else:
        data_np = np.asarray(data)
        
    if labels_is_tensor:
        labels_np = labels.numpy()
    else:
        labels_np = np.asarray(labels)
    
    # Flatten labels if needed
    if len(labels_np.shape) > 1:
        labels_np = labels_np.flatten()
    
    # Find samples with at least one non-missing value
    if len(data_np.shape) == 2:  # (N, D)
        valid_mask = np.any(data_np != missing_value, axis=1)
    elif len(data_np.shape) == 3:  # (N, K, D)
        # Check if all K×D values are missing
        valid_mask = np.any(data_np != missing_value, axis=(1, 2))
    else:
        raise ValueError(f"Unexpected data shape: {data_np.shape}")
    
    filtered_data = data_np[valid_mask]
    filtered_labels = labels_np[valid_mask]
    
    # Convert back to original type
    if data_is_tensor:
        return torch.from_numpy(filtered_data), torch.from_numpy(filtered_labels)
    return filtered_data, filtered_labels

"""Evaluation metrics for RBM models.

This module provides metrics for evaluating RBM predictions, with proper
handling of the label permutation problem via Hungarian algorithm.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .hungarian import get_hungarian_solution, vectorize_predictions


def accuracy_with_hungarian(
    predictions: Union[torch.Tensor, np.ndarray],
    true_labels: Union[torch.Tensor, np.ndarray],
    k: int,
    return_mapping: bool = False,
) -> Union[float, Tuple[float, Dict[int, int]]]:
    """Compute accuracy with Hungarian alignment.

    This function handles the label permutation problem by first finding
    the optimal mapping between predicted and true labels using the
    Hungarian algorithm, then computing accuracy with aligned predictions.

    Parameters
    ----------
    predictions : torch.Tensor or np.ndarray
        Predicted class labels, shape (n_samples,) or (n_samples, 1).
    true_labels : torch.Tensor or np.ndarray
        True class labels, shape (n_samples,) or (n_samples, 1).
    k : int
        Number of classes.
    return_mapping : bool, optional
        If True, also return the class mapping. Default is False.

    Returns
    -------
    float or tuple
        If return_mapping is False: accuracy (float between 0 and 1).
        If return_mapping is True: (accuracy, class_map).

    Examples
    --------
    >>> preds = np.array([2, 2, 1, 1, 0, 0])  # Permuted predictions
    >>> true = np.array([0, 0, 1, 1, 2, 2])   # True labels
    >>> accuracy = accuracy_with_hungarian(preds, true, k=3)
    >>> accuracy
    1.0
    """
    # Get optimal label mapping
    class_map = get_hungarian_solution(predictions, true_labels, k)

    # Apply mapping to predictions
    aligned_preds = vectorize_predictions(predictions, class_map)

    # Convert true labels to numpy for comparison
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    # Compute accuracy
    accuracy = float(np.mean(aligned_preds == true_labels.squeeze()))

    if return_mapping:
        return accuracy, class_map
    return accuracy


def extract_dead_classes_indices(
    conf_matrix: np.ndarray,
    class_map: Dict[int, int],
    dead_threshold: Optional[float] = None,
) -> List[int]:
    """Identify dead (collapsed) classes from confusion matrix.

    A "dead" class is one that the model rarely or never predicts,
    indicating potential mode collapse during training.

    Parameters
    ----------
    conf_matrix : np.ndarray
        Confusion matrix of shape (k, k).
    class_map : dict
        Mapping from predicted class IDs to aligned class IDs.
    dead_threshold : float, optional
        Threshold below which a class is considered dead.
        If None, uses 10% of average class frequency.

    Returns
    -------
    list
        List of RBM class indices that are dead (mapped back via class_map).

    Examples
    --------
    >>> conf = np.array([[100, 0, 0], [0, 100, 0], [0, 5, 0]])  # Class 2 is dead
    >>> class_map = {0: 0, 1: 1, 2: 2}
    >>> dead = extract_dead_classes_indices(conf, class_map)
    >>> 2 in dead
    True
    """
    n_examples = np.sum(conf_matrix)
    predicted_classes_sum = conf_matrix.sum(axis=0)

    if dead_threshold is None:
        dead_threshold = (n_examples / conf_matrix.shape[0]) * 0.1

    dead_classes_indices = np.where(predicted_classes_sum < dead_threshold)[0].tolist()

    # Map back to RBM class indices
    # class_map: {rbm_idx: aligned_idx}, we need reverse mapping
    reverse_map = {v: k for k, v in class_map.items()}
    rbm_dead_indices = [reverse_map[idx] for idx in dead_classes_indices if idx in reverse_map]

    return rbm_dead_indices


def compute_mean_std(
    accuracies: List[float],
    verbose: bool = False,
) -> Tuple[float, float]:
    """Compute mean and standard deviation of accuracies.

    Parameters
    ----------
    accuracies : list
        List of accuracy values (floats between 0 and 1).
    verbose : bool, optional
        If True, print the results. Default is False.

    Returns
    -------
    tuple
        (mean, std) of the accuracies.

    Examples
    --------
    >>> accs = [0.90, 0.92, 0.88, 0.91, 0.89]
    >>> mean, std = compute_mean_std(accs)
    >>> 0.89 < mean < 0.92
    True
    """
    mean = float(np.mean(accuracies))
    std = float(np.std(accuracies))

    if verbose:
        print(f"Mean accuracy: {mean:.4f} ± {std:.4f}")
        print(f"Mean accuracy * 100: {mean * 100:.4f} ± {std * 100:.4f}")

    return mean, std

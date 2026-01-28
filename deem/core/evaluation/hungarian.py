"""Hungarian algorithm for solving label permutation problem.

RBM hidden units have no inherent ordering, so predictions may be permuted
relative to true labels. The Hungarian algorithm finds the optimal mapping
between predicted and true labels to maximize accuracy.

CRITICAL: Always call get_hungarian_solution() before computing accuracy metrics.
"""

from typing import Dict, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _get_class_indices(arr: torch.Tensor, class_id: int) -> torch.Tensor:
    """Get indices of samples belonging to a specific class.

    Parameters
    ----------
    arr : torch.Tensor
        1D tensor of class labels.
    class_id : int
        The class to find indices for.

    Returns
    -------
    torch.Tensor
        Indices where arr == class_id.
    """
    return (arr == class_id).nonzero().squeeze(1)


def _non_intersection(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple:
    """Compute symmetric difference and intersection of two index tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        First set of indices.
    tensor2 : torch.Tensor
        Second set of indices.

    Returns
    -------
    tuple
        (difference, intersection) tensors.
    """
    combined = torch.cat((tensor1, tensor2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return difference, intersection


def build_cost_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> np.ndarray:
    """Build cost matrix for Hungarian algorithm.

    The cost is the number of samples that differ between predicted and
    true class assignments. Lower cost means better alignment.

    Parameters
    ----------
    predictions : torch.Tensor
        Predicted class labels, shape (n_samples,).
    targets : torch.Tensor
        True class labels, shape (n_samples,).
    k : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Cost matrix of shape (k, k) where cost[i,j] is the disagreement
        count when mapping predicted class i to true class j.
    """
    pred_class_indices = [_get_class_indices(predictions, c) for c in range(k)]
    target_class_indices = [_get_class_indices(targets, c) for c in range(k)]

    cost_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            diff, _ = _non_intersection(pred_class_indices[i], target_class_indices[j])
            cost_matrix[i, j] = diff.size(0)

    return cost_matrix


def get_hungarian_solution(
    predictions: Union[torch.Tensor, np.ndarray],
    true_labels: Union[torch.Tensor, np.ndarray],
    k: int,
) -> Dict[int, int]:
    """Solve label permutation problem using Hungarian algorithm.

    CRITICAL: Always call this before computing accuracy.
    RBM hidden units have no inherent ordering, so predictions may use
    a different label assignment than the true labels.

    This function finds the optimal mapping from predicted labels to
    true labels that minimizes classification error.

    Parameters
    ----------
    predictions : torch.Tensor or np.ndarray
        Predicted class labels, shape (n_samples,) or (n_samples, 1).
    true_labels : torch.Tensor or np.ndarray
        True class labels to align against (typically majority vote on
        training set), shape (n_samples,) or (n_samples, 1).
    k : int
        Number of classes.

    Returns
    -------
    dict
        Mapping from predicted class IDs to aligned class IDs.
        Example: {0: 2, 1: 0, 2: 1} means predicted class 0 maps to true class 2.

    Examples
    --------
    >>> preds = torch.tensor([2, 2, 1, 1, 0, 0])  # Permuted predictions
    >>> true = torch.tensor([0, 0, 1, 1, 2, 2])   # True labels
    >>> class_map = get_hungarian_solution(preds, true, k=3)
    >>> class_map
    {0: 2, 1: 1, 2: 0}
    >>> aligned = vectorize_predictions(preds, class_map)
    >>> (aligned == true.numpy()).all()
    True
    """
    # Convert to torch tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(true_labels, np.ndarray):
        true_labels = torch.from_numpy(true_labels)

    # Ensure on CPU and squeeze extra dimensions
    predictions = predictions.cpu().squeeze()
    true_labels = true_labels.cpu().squeeze()

    # Build cost matrix (disagreement counts)
    cost_matrix = build_cost_matrix(predictions, true_labels, k)

    # Solve assignment problem (minimize cost = maximize agreement)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return dict(zip(row_ind, col_ind))


def vectorize_predictions(
    predictions: Union[torch.Tensor, np.ndarray],
    class_map: Dict[int, int],
) -> np.ndarray:
    """Apply Hungarian mapping to predictions.

    Parameters
    ----------
    predictions : torch.Tensor or np.ndarray
        Predicted class labels to transform.
    class_map : dict
        Mapping from predicted class IDs to aligned class IDs,
        as returned by get_hungarian_solution().

    Returns
    -------
    np.ndarray
        Aligned predictions using the class mapping.

    Examples
    --------
    >>> preds = torch.tensor([2, 2, 1, 1, 0, 0])
    >>> class_map = {0: 2, 1: 1, 2: 0}
    >>> aligned = vectorize_predictions(preds, class_map)
    >>> aligned
    array([0, 0, 1, 1, 2, 2])
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    return np.vectorize(class_map.get)(predictions)

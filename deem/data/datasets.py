"""Custom dataset classes for ensemble predictions."""

from torch import Tensor
from torch.utils.data import TensorDataset


class TensorDatasetWithShape(TensorDataset):
    """
    TensorDataset that exposes the shape of the first tensor.
    
    Used to detect soft labels: if shape has 3 dimensions (N, K, D),
    we know it's probability distributions. If 2 dimensions (N, D),
    it's hard labels.
    
    Parameters
    ----------
    *tensors : Tensor
        Tensors to store in the dataset. First tensor should be predictions.
        
    Examples
    --------
    >>> import torch
    >>> # Hard labels
    >>> hard = TensorDatasetWithShape(
    ...     torch.randint(0, 3, (100, 15)),  # (N, D)
    ...     torch.randint(0, 3, (100,))       # labels
    ... )
    >>> assert len(hard.shape) == 2  # 2D = hard labels
    
    >>> # Soft labels  
    >>> soft = TensorDatasetWithShape(
    ...     torch.rand(100, 3, 15),  # (N, K, D)
    ...     torch.randint(0, 3, (100,))
    ... )
    >>> assert len(soft.shape) == 3  # 3D = soft labels
    """
    
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__(*tensors)
        self.shape = self.tensors[0].shape
    
    def __repr__(self) -> str:
        return f"TensorDatasetWithShape(shape={self.shape}, n_samples={len(self)})"

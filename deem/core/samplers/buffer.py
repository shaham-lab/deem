"""Persistent replay buffer for MCMC chain management.

This module provides a buffer for storing and managing samples during
RBM training with persistent contrastive divergence (PCD).
"""

import torch
from typing import List, Optional, Tuple, Union


class Buffer:
    """A buffer class for managing persistent MCMC chains.

    Supports populating with examples, automatic size management,
    and random sampling for training.

    Attributes:
        buffer_max_size: The maximum number of examples to store.
        examples: List of tensor examples in the buffer.

    Example:
        >>> buffer = Buffer(max_size=1000)
        >>> buffer.add_examples(torch.randn(64, 3, 10))
        >>> samples = buffer.get_random_examples(32)
        >>> print(samples.shape)
        torch.Size([32, 3, 10])
    """

    def __init__(self, buffer_max_size: int):
        """Initialize the buffer with a maximum size.

        Args:
            buffer_max_size: Maximum number of examples to keep.
        """
        self.buffer_max_size = buffer_max_size
        self.examples: List[torch.Tensor] = []

    def add_examples(self, examples: torch.Tensor) -> None:
        """Add examples to the buffer.

        If buffer exceeds max size, oldest examples are removed.

        Args:
            examples: Tensor of shape (batch_size, ...) to add.
        """
        # Extend list with new examples
        self.examples.extend(torch.unbind(examples, dim=0))
        # Keep only most recent examples up to max size
        if len(self.examples) > self.buffer_max_size:
            self.examples = self.examples[-self.buffer_max_size:]

    def get_random_examples(
        self, 
        num_examples: Optional[int] = None,
        with_indices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """Get random examples from the buffer.

        Args:
            num_examples: Number of examples to retrieve. If None, returns all.
            with_indices: If True, also return the indices used.

        Returns:
            Tensor of examples, or tuple of (tensor, indices) if with_indices.
        """
        if len(self.examples) == 0:
            raise ValueError("Buffer is empty, cannot get examples")
            
        if num_examples is None or num_examples >= len(self.examples):
            tensor = torch.stack(self.examples)
            if with_indices:
                return tensor, list(range(len(self.examples)))
            return tensor
            
        indices = torch.randperm(len(self.examples))[:num_examples].tolist()
        tensor = torch.stack([self.examples[i] for i in indices])
        
        if with_indices:
            return tensor, indices
        return tensor

    def get_random_examples_with_indices(
        self, 
        num_examples: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """Get random examples with their indices.

        Args:
            num_examples: Number of examples to retrieve.

        Returns:
            Tuple of (tensor of examples, list of indices).
        """
        return self.get_random_examples(num_examples, with_indices=True)

    def modify_by_indices(self, data: torch.Tensor, indices: Union[int, List[int]]) -> None:
        """Modify examples at specified indices.

        Args:
            data: Tensor of new data to insert.
            indices: Indices where to insert the data.

        Raises:
            ValueError: If number of indices doesn't match batch size.
            IndexError: If any index is out of range.
        """
        if not isinstance(indices, list):
            indices = [indices]

        if len(indices) != data.size(0):
            raise ValueError("Number of indices must match the batch size of data")

        for i, index in enumerate(indices):
            if index < 0 or index >= len(self.examples):
                raise IndexError(f"Index {index} is out of range for examples list")
            self.examples[index] = data[i]

    def clear(self) -> None:
        """Clear all examples from the buffer."""
        self.examples = []

    def __getitem__(
        self, 
        indices: Union[int, slice, List[int], torch.Tensor]
    ) -> torch.Tensor:
        """Index the buffer like a list or tensor.

        Args:
            indices: Integer, slice, list, or tensor of indices.

        Returns:
            Single tensor or stacked tensor of examples.
        """
        if isinstance(indices, int):
            return self.examples[indices]
        if isinstance(indices, slice):
            return torch.stack(self.examples[indices])
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return torch.stack([self.examples[i] for i in indices])

    def __len__(self) -> int:
        """Return the number of examples in the buffer."""
        return len(self.examples)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Buffer(size={len(self)}, max_size={self.buffer_max_size})"

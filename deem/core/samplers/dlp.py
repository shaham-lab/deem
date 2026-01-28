"""Diffusion Langevin Proposal (DLP) sampler for RBMs.

This module implements the DLP sampler which uses continuous relaxation
and Langevin dynamics to propose moves in discrete spaces.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from .buffer import Buffer


# Memory limit for safe einsum operations (GB)
MEMORY_LIMIT_GB = 2.0


def _revert_one_hot(one_hot_tensor: torch.Tensor) -> torch.Tensor:
    """Convert one-hot encoded tensor back to class indices.
    
    Args:
        one_hot_tensor: Tensor of shape (batch, k, dx) with one-hot encoding.
        
    Returns:
        Tensor of shape (batch, dx) with class indices.
    """
    return one_hot_tensor.argmax(dim=1)


@runtime_checkable
class DlpModel(Protocol):
    """Protocol for models compatible with DLP sampler.
    
    Models must provide:
    - device: The device tensors should be on
    - l: Number of classes/categories (k)
    - preprocess: Convert 2D indices to one-hot representation
    - forward: Compute visible, intermediate, and hidden states
    - negative_energy: Compute negative energy for states
    """
    device: torch.device
    l: int
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to one-hot representation."""
        ...
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (visible, intermediate, hidden)."""
        ...
    
    def negative_energy(self, intermediate: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute negative energy."""
        ...


class DlpSampler:
    """Diffusion Langevin Proposal sampler for multinomial RBMs.

    Uses continuous relaxation with Langevin dynamics to propose moves,
    then discretizes with Metropolis-Hastings acceptance.

    Attributes:
        model: The RBM model to sample from.
        input_shape: Shape of the input tensor.
        step_size: Step size for Langevin dynamics.
        max_len: Maximum buffer size for persistent chains.
        buffer: Replay buffer for chain management.
        oh_mode: Whether inputs are one-hot encoded.
        eye_tensor: Precomputed identity tensor for efficiency.

    Example:
        >>> sampler = DlpSampler(model, input_shape=(3, 15), step_size=0.2)
        >>> fake_samples = sampler.sample_new_exmps(real_data, steps=10)
    """

    def __init__(
        self,
        model: DlpModel,
        input_shape: Tuple[int, ...],
        step_size: float = 0.2,
        max_len: int = 8192,
        oh_mode: bool = False
    ):
        """Initialize the DLP sampler.

        Args:
            model: The RBM model for computing energies.
            input_shape: Shape of input (k, dx) or (l, i).
            step_size: Step size for proposal distribution.
            max_len: Maximum examples in replay buffer.
            oh_mode: If True, inputs are one-hot encoded.
        """
        self.model = model
        self.input_shape = input_shape
        self.max_len = max_len
        self.buffer = Buffer(max_len)
        self.step_size = step_size
        self.oh_mode = oh_mode
        
        # Precompute identity tensor for efficiency
        self.eye_tensor = (
            torch.eye(model.l)
            .unsqueeze(0)
            .unsqueeze(3)
            .to(model.device)
        )  # Shape: (1, L, L, 1)

    def sample_new_exmps(
        self, 
        inputs: torch.Tensor, 
        steps: int = 60
    ) -> torch.Tensor:
        """Generate new "fake" samples for contrastive divergence.

        Uses 95% buffer samples + 5% new samples, then runs MCMC.
        When buffer is empty (first iteration), uses all inputs.

        Args:
            inputs: Real data tensor to use for initializing new chains.
            steps: Number of MCMC steps.

        Returns:
            Tensor of generated samples.
        """
        n = inputs.size(0)
        
        # When buffer is empty, use all inputs to bootstrap
        if len(self.buffer) == 0:
            inp_inputs = inputs.clone().detach().to(self.model.device)
        else:
            # Choose 95% from buffer, 5% new from real data
            n_new = np.random.binomial(n, 0.05)
            n_new = max(n_new, 1)  # At least 1 new sample
            
            buffer_inputs = self.buffer.get_random_examples(n - n_new)
            
            # Get random subset of real data for new chains
            batch_rand_indices = torch.randperm(n)[:n_new]
            batch_inputs = inputs[batch_rand_indices]

            # Combine buffer and new samples
            inp_inputs = torch.cat(
                [batch_inputs.to('cpu'), buffer_inputs], dim=0
            ).detach().to(self.model.device)

        # Run MCMC sampling
        inp_inputs = self.generate_samples(self.model, inp_inputs, steps=steps)

        # Update buffer
        self.buffer.add_examples(inp_inputs.cpu())

        return inp_inputs

    def generate_samples(
        self,
        model: DlpModel,
        inp_inputs: torch.Tensor,
        steps: int = 60,
        return_input_per_step: bool = False
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Generate samples using DLP Metropolis-Hastings.

        Args:
            model: The RBM model.
            inp_inputs: Initial samples to start chains from.
            steps: Number of MCMC steps.
            return_input_per_step: If True, return samples at each step.

        Returns:
            Generated samples, or tensor of samples at each step.
        """
        with torch.set_grad_enabled(True):
            is_training = model.training
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            inputs_per_step = []
            
            # Convert from one-hot if needed
            if self.oh_mode:
                inp_inputs = _revert_one_hot(inp_inputs).float()
                
            N, I = inp_inputs.shape[0], inp_inputs.shape[-1]

            # Pre-allocate batch indices
            batch_indices = (
                torch.arange(N)
                .unsqueeze(-1)
                .repeat(1, I)
                .to(inp_inputs.device)
            )

            for _ in range(steps):
                # Forward pass with current samples
                oh_inputs = model.preprocess(inp_inputs)
                oh_inputs.requires_grad = True
                
                oh_visible, oh_intermediate, oh_hidden = model(oh_inputs)
                neg_energy = model.negative_energy(oh_intermediate, oh_hidden)
                neg_energy.sum().backward()

                # Calculate transition probabilities
                q_i = self._calculate_q(oh_visible)
                sampled_indices = torch.multinomial(
                    q_i.permute(0, 2, 1).reshape(-1, q_i.size(1)),
                    num_samples=1
                ).reshape(N, I)

                # Forward pass with proposed samples
                oh_suggested_inputs = model.preprocess(sampled_indices.float())
                oh_suggested_inputs.requires_grad = True
                
                oh_visible_suggested, oh_intermediate_suggested, oh_hidden_suggested = model(
                    oh_suggested_inputs
                )
                neg_energy_suggested = model.negative_energy(
                    oh_intermediate_suggested, oh_hidden_suggested
                )
                neg_energy_suggested.sum().backward()

                # Calculate reverse transition probabilities
                q_i_suggested = self._calculate_q(oh_visible_suggested)

                # Compute acceptance probability
                exponent_grad_diff = torch.exp(neg_energy_suggested - neg_energy)
                i_range = torch.arange(I).to(inp_inputs.device)
                
                q_real_given_suggested = q_i[
                    batch_indices, sampled_indices, i_range
                ].prod(1)
                q_suggested_given_real = q_i_suggested[
                    batch_indices, inp_inputs.long(), i_range
                ].prod(1)
                
                accept_probabilities = (
                    exponent_grad_diff 
                    * (q_real_given_suggested / q_suggested_given_real)
                ).clamp_(min=1)

                # Accept/reject proposed changes
                accepted = torch.rand_like(accept_probabilities) <= accept_probabilities
                inp_inputs[accepted] = sampled_indices[accepted].float()

                if return_input_per_step:
                    inputs_per_step.append(inp_inputs.clone())

            # Restore model state
            for p in model.parameters():
                p.requires_grad = True
            model.train(is_training)

            # Convert back to one-hot if needed
            if self.oh_mode:
                inp_inputs = model.preprocess(inp_inputs)

            if return_input_per_step:
                return torch.stack(inputs_per_step, dim=0)
            return inp_inputs

    def _calculate_q(self, inp_inputs: torch.Tensor) -> torch.Tensor:
        """Calculate the proposal distribution.

        Uses memory-safe computation for large inputs.

        Args:
            inp_inputs: One-hot encoded inputs with gradients.

        Returns:
            Proposal distribution tensor.
        """
        inp_detached = inp_inputs.detach()  # (n, l, i)
        delta_f = inp_inputs.grad.detach()

        # Estimate memory usage for clf_diff: (n, l, l, i)
        n, l, i = inp_detached.shape
        estimated_gb = n * l * l * i * 4 / (1024 ** 3)  # float32 size

        if estimated_gb <= MEMORY_LIMIT_GB:
            return self._calculate_q_regular(inp_detached, delta_f)
        else:
            return self._calculate_q_safe(inp_detached, delta_f)

    def _calculate_q_regular(
        self, 
        inp_detached: torch.Tensor, 
        delta_f: torch.Tensor
    ) -> torch.Tensor:
        """Calculate q distribution (memory-intensive version).

        Args:
            inp_detached: Detached input tensor.
            delta_f: Gradient tensor.

        Returns:
            Proposal distribution.
        """
        n_chunks, chunks = 15, []
        normalization_chunks = []
        
        for delta_f_chunk, inp_detached_chunk in zip(
            delta_f.chunk(n_chunks, dim=0), 
            inp_detached.chunk(n_chunks, dim=0)
        ):
            clf_diff_chunk = self.eye_tensor - inp_detached_chunk.unsqueeze(2)
            chunks.append(
                torch.einsum('nli,nlki->nki', delta_f_chunk, clf_diff_chunk)
            )
            norm_chunk = (
                torch.linalg.norm(clf_diff_chunk, ord=2, dim=1) ** 2 
                / (2 * self.step_size)
            )
            normalization_chunks.append(norm_chunk)

        gradient_term = 0.5 * torch.cat(chunks, dim=0)
        normalization_term = torch.cat(normalization_chunks, dim=0)
        d_x = gradient_term - normalization_term
        return F.softmax(d_x, dim=1)

    def _calculate_q_safe(
        self, 
        inp_detached: torch.Tensor, 
        delta_f: torch.Tensor
    ) -> torch.Tensor:
        """Calculate q distribution (memory-safe version).

        Args:
            inp_detached: Detached input tensor.
            delta_f: Gradient tensor.

        Returns:
            Proposal distribution.
        """
        n, l, i = inp_detached.shape
        per_sample_bytes = l * l * i * 4
        max_chunk_size = max(1, int((MEMORY_LIMIT_GB * (1024 ** 3)) // per_sample_bytes))

        grad_chunks = []
        norm_chunks = []

        for start in range(0, n, max_chunk_size):
            end = min(n, start + max_chunk_size)
            inp_chunk = inp_detached[start:end]
            grad_chunk = delta_f[start:end]

            clf_diff = self.eye_tensor - inp_chunk.unsqueeze(2)
            grad_term_chunk = 0.5 * torch.einsum(
                'bli,blki->bki', grad_chunk, clf_diff
            )
            norm_term_chunk = (
                torch.linalg.norm(clf_diff, ord=2, dim=1) ** 2 
                / (2 * self.step_size)
            )

            grad_chunks.append(grad_term_chunk)
            norm_chunks.append(norm_term_chunk)

        gradient_term = torch.cat(grad_chunks, dim=0)
        normalization_term = torch.cat(norm_chunks, dim=0)
        d_x = gradient_term - normalization_term
        return F.softmax(d_x, dim=1)

    def clear_buffer(self) -> None:
        """Clear the replay buffer."""
        self.buffer.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DlpSampler(input_shape={self.input_shape}, "
            f"step_size={self.step_size}, max_len={self.max_len}, "
            f"oh_mode={self.oh_mode})"
        )

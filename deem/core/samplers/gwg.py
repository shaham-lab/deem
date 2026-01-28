"""Gibbs with Gradients (GWG) sampler for RBMs.

This module implements the GWG sampler which uses gradient information
to propose efficient Metropolis-Hastings moves in discrete spaces.

Reference: Grathwohl et al., "Oops I Took A Gradient: Scalable Sampling
for Discrete Distributions" (2021)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from .buffer import Buffer


@runtime_checkable
class GwgModel(Protocol):
    """Protocol for models compatible with GWG sampler.
    
    Models must provide:
    - device: The device tensors should be on
    - preprocess: Convert 2D indices to one-hot representation
    - forward: Compute visible, intermediate, and hidden states
    - negative_energy: Compute negative energy for states
    """
    device: torch.device
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to one-hot representation."""
        ...
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (visible, intermediate, hidden)."""
        ...
    
    def negative_energy(self, intermediate: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute negative energy."""
        ...


class GwgSampler:
    """Gibbs with Gradients sampler for multinomial RBMs.

    Uses gradient information to propose efficient moves in the discrete
    input space, with Metropolis-Hastings acceptance.

    Attributes:
        model: The RBM model to sample from.
        input_shape: Shape of the input tensor (k, dx).
        max_len: Maximum buffer size for persistent chains.
        buffer: Replay buffer for chain management.
        oh_mode: Whether inputs are already one-hot encoded.

    Example:
        >>> sampler = GwgSampler(model, input_shape=(3, 15), max_len=8192)
        >>> fake_samples = sampler.sample_new_exmps(real_data, steps=10)
    """

    def __init__(
        self,
        model: GwgModel,
        input_shape: Tuple[int, ...],
        max_len: int = 8192,
        oh_mode: bool = False
    ):
        """Initialize the GWG sampler.

        Args:
            model: The RBM model for computing energies.
            input_shape: Shape of input (k, dx) or (l, i).
            max_len: Maximum examples in replay buffer.
            oh_mode: If True, inputs are one-hot encoded.
        """
        self.model = model
        self.input_shape = input_shape
        self.max_len = max_len
        self.buffer = Buffer(max_len)
        self.oh_mode = oh_mode

    def sample_new_exmps(
        self, 
        inputs: torch.Tensor, 
        steps: int = 60
    ) -> torch.Tensor:
        """Generate new "fake" samples for contrastive divergence.

        Uses 95% buffer samples + 5% new samples, then runs MCMC.

        Args:
            inputs: Real data tensor to use for initializing new chains.
            steps: Number of MCMC steps.

        Returns:
            Tensor of generated samples.
        """
        n = inputs.size(0)
        n_new = np.random.binomial(n, 0.05)

        # Get samples from buffer
        if len(self.buffer) > 0:
            buffer_inputs = self.buffer.get_random_examples(n - n_new)
        else:
            buffer_inputs = torch.empty(0, *inputs.shape[1:])

        # Get random subset of real data for new chains
        batch_rand_indices = torch.randperm(n)[:n_new]
        batch_inputs = inputs[batch_rand_indices]

        # Combine buffer and new samples
        inp_inputs = torch.cat(
            [batch_inputs.to('cpu'), buffer_inputs], dim=0
        ).detach().to(self.model.device)

        # Run MCMC sampling
        inp_inputs = self.generate_samples(self.model, inp_inputs, steps=steps)

        # Update buffer (only during evaluation to avoid training interference)
        if not self.model.training:
            self.buffer.add_examples(inp_inputs.cpu())

        return inp_inputs

    def generate_samples(
        self,
        model: GwgModel,
        inp_inputs: torch.Tensor,
        steps: int = 60,
        return_input_per_step: bool = False
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Generate samples using GWG Metropolis-Hastings.

        Args:
            model: The RBM model.
            inp_inputs: Initial samples to start chains from.
            steps: Number of MCMC steps.
            return_input_per_step: If True, return samples at each step.

        Returns:
            Generated samples, or tensor of samples at each step.
        """
        # Disable gradients for model parameters during sampling
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        inputs_per_step = []

        for _ in range(steps):
            # Forward pass with current samples
            oh_inputs = model.preprocess(inp_inputs)
            oh_inputs.requires_grad = True

            oh_visible, oh_intermediate, oh_hidden = model(oh_inputs)
            neg_energy = model.negative_energy(oh_intermediate, oh_hidden)
            neg_energy.sum().backward()

            # Calculate proposal distribution q(i|x)
            q_x = self._calculate_q(oh_visible)
            sampled_indices = torch.multinomial(q_x, num_samples=1).squeeze(-1)

            # Flip dimension to get proposed sample
            suggested_inputs_2dim = self._flip_dim_2dim(inp_inputs, sampled_indices)

            # Forward pass with proposed samples
            oh_suggested_inputs = model.preprocess(suggested_inputs_2dim)
            oh_suggested_inputs.requires_grad = True

            oh_visible_suggested, oh_intermediate_suggested, oh_hidden_suggested = model(
                oh_suggested_inputs
            )
            neg_energy_suggested = model.negative_energy(
                oh_intermediate_suggested, oh_hidden_suggested
            )
            neg_energy_suggested.sum().backward()

            # Calculate reverse proposal q(i|x')
            q_x_suggested = self._calculate_q(oh_visible_suggested)

            # Compute acceptance probability
            exponent_grad_diff = torch.exp(neg_energy_suggested - neg_energy)
            n_indices = torch.arange(inp_inputs.size(0))
            q_x_div = (
                q_x_suggested[n_indices, sampled_indices] 
                / q_x[n_indices, sampled_indices]
            )
            accept_probabilities = (exponent_grad_diff * q_x_div).clamp_(min=1)

            # Accept/reject with computed probabilities
            inp_inputs = self._flip_dim_2dim(
                inp_inputs, sampled_indices, accept_probabilities=accept_probabilities
            )

            if return_input_per_step:
                inputs_per_step.append(inp_inputs.clone().detach())

        # Restore model state
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if return_input_per_step:
            return torch.stack(inputs_per_step, dim=0)
        return inp_inputs

    def _calculate_q(self, inp_inputs: torch.Tensor) -> torch.Tensor:
        """Calculate the proposal distribution q(i|x).

        Uses gradients to bias proposals toward lower energy states.

        Args:
            inp_inputs: One-hot encoded inputs with gradients.

        Returns:
            Proposal distribution tensor of shape (batch, l*dx).
        """
        delta_f = inp_inputs.grad.data.detach_()  # N x L x I
        _, l, dx = inp_inputs.shape
        
        # Compute difference in gradients
        d_x = delta_f - torch.einsum(
            'nli,nli->ni', inp_inputs, delta_f
        ).unsqueeze(1)
        
        # Softmax over flattened dimensions
        q_x = F.softmax(d_x.reshape(d_x.size(0), l * dx) / 2, dim=-1)
        return q_x

    def _flip_dim_2dim(
        self,
        inp_inputs: torch.Tensor,
        sampled_indices: torch.Tensor,
        accept_probabilities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flip a dimension in the 2D input based on sampled indices.

        Args:
            inp_inputs: Input tensor of shape (N, I).
            sampled_indices: Indices to flip.
            accept_probabilities: Optional acceptance probabilities.

        Returns:
            Updated input tensor with flipped dimensions.
        """
        N, I = inp_inputs.size()
        
        if accept_probabilities is None:
            accept_probabilities = torch.ones(N).float()

        inp_inputs = inp_inputs.clone()

        # Accept/reject based on probabilities
        accepted_mask = (
            torch.rand_like(accept_probabilities) <= accept_probabilities
        ).to('cpu')
        accepted_indices = sampled_indices[accepted_mask]

        # Calculate reverse indices
        n_indices = torch.arange(N)[accepted_mask]
        dx_indices = accepted_indices % I
        values = accepted_indices // I

        inp_inputs[n_indices, dx_indices] = values.float()

        return inp_inputs

    def clear_buffer(self) -> None:
        """Clear the replay buffer."""
        self.buffer.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GwgSampler(input_shape={self.input_shape}, "
            f"max_len={self.max_len}, oh_mode={self.oh_mode})"
        )

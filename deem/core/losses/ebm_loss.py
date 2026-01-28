"""Energy-based model loss functions for RBM training.

This module provides loss functions for training RBMs using contrastive divergence.
The losses compute the difference between positive phase (data) and negative phase
(model samples) energies.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class EnergyModel(Protocol):
    """Protocol for models that can compute energy."""

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute energy for visible-hidden configurations.

        Parameters
        ----------
        v : torch.Tensor
            Visible unit states.
        h : torch.Tensor
            Hidden unit states.

        Returns
        -------
        torch.Tensor
            Energy values for each sample in the batch.
        """
        ...


class EBMLoss(nn.Module):
    """Energy-based model loss function for contrastive divergence training.

    This loss implements the standard CD objective:
        L = E[E(v_pos, h_pos)] - E[E(v_neg, h_neg)]

    Where E is the energy function, v_pos/h_pos are data samples, and
    v_neg/h_neg are model samples from CD.

    Parameters
    ----------
    model : nn.Module
        The energy-based model (must have an `energy` method).
    with_norm : bool, optional
        If True, adds energy regularization term to prevent unbounded energies.
        Default is False.

    Examples
    --------
    >>> model = MultinomialRBM(dx=15, dh=1, k=3)
    >>> loss_fn = EBMLoss(model)
    >>> v_pos, h_pos, v_neg, h_neg = model.contrastive_divergence(data)
    >>> loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
    >>> loss.backward()
    """

    def __init__(self, model: nn.Module, with_norm: bool = False):
        """Initialize the EBMLoss module.

        Parameters
        ----------
        model : nn.Module
            The energy-based model for which the loss will be computed.
            Must have an `energy(v, h)` method.
        with_norm : bool, optional
            Whether to add energy regularization. Default is False.
        """
        super().__init__()
        self.model = model
        self.with_norm = with_norm

    def forward(
        self,
        v_pos: torch.Tensor,
        h_pos: torch.Tensor,
        v_neg: torch.Tensor,
        h_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the EBM loss.

        Parameters
        ----------
        v_pos : torch.Tensor
            Positive phase visible units (data samples).
        h_pos : torch.Tensor
            Positive phase hidden units (from data).
        v_neg : torch.Tensor
            Negative phase visible units (model samples).
        h_neg : torch.Tensor
            Negative phase hidden units (from model).

        Returns
        -------
        torch.Tensor
            Scalar loss value. Minimizing this maximizes the log-likelihood.
        """
        pos_energy = self.model.energy(v_pos, h_pos)
        neg_energy = self.model.energy(v_neg, h_neg)

        # Log-likelihood: log p(v) ∝ -E(v_pos) + E(v_neg) (free energy difference)
        # We maximize log-likelihood -> minimize negative log-likelihood
        loglik = -pos_energy + neg_energy
        loss = -loglik.mean()

        if self.with_norm:
            # Regularize to prevent energies from growing unbounded
            loss = loss + (pos_energy**2 + neg_energy**2).mean()

        return loss


# Alias for clearer naming
ContrastiveDivergenceLoss = EBMLoss

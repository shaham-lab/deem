"""Multinomial RBM with Gibbs-with-Gradients sampler.

This module implements the production MultinomialRBMGwg model which combines
the base MultinomialRBM with DLP/GWG sampling for contrastive divergence training.

This is THE production model used in run_predict.py.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MultinomialRBM
from ..samplers.gwg import GwgSampler
from ..samplers.dlp import DlpSampler
from ..preprocessing import Multinomial


class MultinomialRBMGwg(MultinomialRBM):
    """Multinomial RBM with DLP/GWG sampler for production use.
    
    This model extends MultinomialRBM with:
    - DLP (Diffusion Langevin Proposal) or GWG sampling for CD training
    - Optional multinomial preprocessing layer
    - Support for both hard and soft labels via preprocess()
    
    The model uses energy-based contrastive divergence training where
    fake samples are generated via MCMC sampling using gradient information.
    
    Parameters
    ----------
    dx : int
        Number of visible units (classifiers/annotators).
    dh : int
        Number of hidden units (learned representation size).
    k : int
        Number of classes for the classification task.
    l : int, optional
        Number of multinomial visible states. Default is k.
    m : int, optional
        Number of multinomial hidden states. Default is k.
    device : torch.device or str
        Device for computations.
    cd_k : int, optional
        Number of Contrastive Divergence steps. Default is 10.
    deterministic : bool, optional
        If True, use probabilities instead of sampling. Default is False.
    sampler_steps : int, optional
        Number of MCMC steps for sampling. Default is 5.
    oh_mode : bool, optional
        Whether inputs are one-hot encoded. Default is False.
    multinomial_net : nn.Module, optional
        Optional preprocessing network. Default is None.
    init_method : str, optional
        Weight initialization: 'rand', 'mv', 'mv_rand', etc. Default is 'rand'.
    **kwargs
        Additional arguments passed to MultinomialRBM base class.
        
    Attributes
    ----------
    sampler : DlpSampler
        The DLP sampler for generating fake samples.
    multinomial_net : nn.Module or None
        Optional preprocessing layer.
    rbm_params : list
        List of RBM parameters (weights and biases).
    multinomial_params : list or None
        Parameters from multinomial_net if present.
        
    Example
    -------
    >>> rbm = MultinomialRBMGwg(
    ...     dx=15, dh=3, k=3, l=3, m=3, 
    ...     device='cuda', cd_k=10, deterministic=True
    ... )
    >>> data = torch.randint(0, 3, (32, 15))  # Hard labels
    >>> visible, intermediate, hidden = rbm(data)
    >>> predictions = rbm.predict(data)
    """
    
    def __init__(self, **kwargs):
        """Initialize MultinomialRBMGwg with sampler and optional preprocessing."""
        # Initialize base MultinomialRBM
        super().__init__(**kwargs)
        
        # Get optional multinomial preprocessing network
        multinomial_net = kwargs.get('multinomial_net', None)
        
        if multinomial_net is not None:
            self.multinomial_net = multinomial_net
            self.multinomial_params = list(multinomial_net.parameters())
        else:
            self.register_parameter('multinomial_net', None)
            self.multinomial_params = None
        
        # Register parameters for optimization
        self.weights = nn.Parameter(self.weights)
        self.visible_bias = nn.Parameter(self.visible_bias)
        self.hidden_bias = nn.Parameter(self.hidden_bias)
        
        # Store RBM parameters for selective training
        self.rbm_params = [self.weights, self.visible_bias, self.hidden_bias]
        
        # Save fixed weights for first hidden unit (if dh==1)
        # This is CRITICAL for matching production behavior
        self.save_fixed_weights()
        
        # Setup sampler
        self.sampler_steps = kwargs.get('sampler_steps', 5)
        oh_mode = kwargs.get('oh_mode', False)
        
        # Use DLP sampler (default production sampler)
        self.sampler = DlpSampler(
            self, 
            input_shape=(self.l, self.dx),
            oh_mode=oh_mode
        )
    
    def apply_multinomial_layer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Apply the optional multinomial preprocessing layer.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input data of shape (N, L, D) after preprocess().
            
        Returns
        -------
        torch.Tensor
            Transformed data, or input_data if no multinomial_net.
        """
        if self.multinomial_net is not None:
            return self.multinomial_net(input_data)
        return input_data
    
    def predict(self, batch: torch.Tensor, as_distribution: bool = False) -> torch.Tensor:
        """Predict output labels with preprocessing.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch (hard or soft labels).
        as_distribution : bool, optional
            If True, sample from distribution; if False, use argmax.
            
        Returns
        -------
        torch.Tensor
            Predicted labels of shape (N, H).
        """
        batch = self.preprocess(batch)
        batch = self.apply_multinomial_layer(batch)
        return super().predict(batch, as_distribution)
    
    def print_energy_metrics(self, input_data: torch.Tensor) -> float:
        """Calculate and print energy metrics with preprocessing.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input data (hard or soft labels).
            
        Returns
        -------
        float
            Sum of energy values.
        """
        input_data = self.preprocess(input_data)
        input_data = self.apply_multinomial_layer(input_data)
        # Call parent's parent (RBM) print_energy_metrics
        h = self.calc_hidden_probs(input_data)
        import numpy as np
        energy = self.energy(input_data, h).detach().cpu().numpy()
        energy_sum = np.sum(energy)
        self.energies_sum.append(energy_sum)
        print(f'Energy - sum:{energy_sum}, mean:{np.mean(energy)}')
        return energy_sum
    
    def forward(
        self, input_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute visible input, intermediate representation, and hidden activations.
        
        The input data is preprocessed, optionally passed through the multinomial
        layer, then hidden probabilities are calculated and sampled.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input data (hard labels (N, D) or soft labels (N, K, D)).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - visible_input_data: Preprocessed input (N, L, D)
            - intermediate_input: After multinomial layer (N, L, D)
            - hidden_activations: Sampled hidden states (N, H, M)
        """
        visible_input_data = self.preprocess(input_data)
        intermediate_input = self.apply_multinomial_layer(visible_input_data)
        
        with torch.no_grad():
            hidden_probabilities = self.calc_hidden_probs(intermediate_input)
            hidden_activations = self.sample_from_hidden_probs(hidden_probabilities)
        
        return visible_input_data, intermediate_input, hidden_activations
    
    def get_samples(
        self, 
        batch: torch.Tensor,
        train_multi: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get real and fake samples for contrastive divergence training.
        
        This is the core training method. It:
        1. Generates fake samples using the DLP sampler
        2. Computes forward pass for both real and fake samples
        3. Optionally freezes/unfreezes multinomial or RBM params
        
        Parameters
        ----------
        batch : torch.Tensor
            Real input data batch.
        train_multi : bool, optional
            If provided, controls which parameters to train:
            - True: Only train multinomial_net params
            - False: Only train RBM params
            - None: Train all parameters
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - real_visible: Intermediate representation of real data
            - real_hidden: Hidden activations for real data
            - fake_visible: Intermediate representation of fake data
            - fake_hidden: Hidden activations for fake data
        """
        # Generate fake samples via MCMC
        fake_inputs = self.sampler.sample_new_exmps(
            inputs=batch,
            steps=self.sampler_steps
        )
        # Handle last batch (may be smaller than sampler output)
        fake_inputs = fake_inputs[:batch.shape[0], :]
        
        # Enable gradients for multinomial_net if present
        if self.multinomial_net is not None:
            for p in self.multinomial_net.parameters():
                p.requires_grad = True
        
        # Handle selective training
        if train_multi is not None:
            if not train_multi:
                # Freeze multinomial params, train RBM
                if self.multinomial_params is not None:
                    for param in self.multinomial_params:
                        param.requires_grad = False
            else:
                # Freeze RBM params, train multinomial
                for param in self.rbm_params:
                    param.requires_grad = False
        
        # Forward pass for real and fake data
        _, real_visible, real_hidden = self.forward(batch)
        _, fake_visible, fake_hidden = self.forward(fake_inputs)
        
        return real_visible, real_hidden, fake_visible, fake_hidden
    
    def negative_energy(
        self, visible: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative energy (log-probability proxy).
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations.
        hidden : torch.Tensor
            Hidden layer activations.
            
        Returns
        -------
        torch.Tensor
            Negative energy values.
        """
        return -self.energy(visible, hidden)


# Backward compatibility alias
RBMGwg = MultinomialRBMGwg

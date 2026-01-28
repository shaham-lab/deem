"""Base classes for Restricted Boltzmann Machines.

This module defines the abstract base classes for all RBM implementations:
- RBM: Abstract base class with core interface methods
- MultinomialRBM: Mixin providing multinomial-specific functionality

The key method is `preprocess()` which handles both hard and soft label inputs:
- Hard labels (N, D): Converted to one-hot encoding (N, K, D)
- Soft labels (N, K, D): Passed through unchanged
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class RBM(nn.Module, ABC):
    """Abstract base class for all RBM implementations.
    
    This class defines the interface that all RBM variants must implement.
    It inherits from both nn.Module for PyTorch integration and ABC for
    abstract method enforcement.
    
    Parameters
    ----------
    dx : int
        Number of visible units (classifiers/annotators).
    dh : int
        Number of hidden units (learned representation size).
    cd_k : int
        Number of Contrastive Divergence steps.
    device : torch.device or str
        Device for computations ('cpu' or 'cuda:X').
    learning_rate : float, optional
        Learning rate for training. Default is 1e-3.
    momentum_coefficient : float, optional
        Momentum coefficient for training. Default is 0.5.
    reg_lambda : float, optional
        Regularization lambda value. Default is 1e-3.
    weight_decay : float, optional
        Weight decay value. Default is 1e-4.
    as_distribution : bool, optional
        If True, sample from distributions; if False, use argmax. Default is True.
    deterministic : bool, optional
        If True, use probabilities instead of sampling during training.
        When True, as_distribution is ignored. Default is False.
    
    Attributes
    ----------
    energies_sum : list
        History of energy sums during training (for monitoring).
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.dx = kwargs.get('dx')
        self.dh = kwargs.get('dh')
        self.cd_k = kwargs.get('cd_k')
        self.device = kwargs.get('device', 'cpu')
        
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.momentum_coefficient = kwargs.get('momentum_coefficient', 0.5)
        
        self.reg_lambda = kwargs.get('reg_lambda', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.as_distribution = kwargs.get('as_distribution', True)
        self.deterministic = kwargs.get('deterministic', False)
        
        self.energies_sum = []
    
    @abstractmethod
    def calc_hidden_probs(self, visible: torch.Tensor) -> torch.Tensor:
        """Calculate hidden layer probabilities from visible layer.
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations.
            
        Returns
        -------
        torch.Tensor
            Hidden layer probabilities.
        """
        pass
    
    @abstractmethod
    def calc_visible_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Calculate visible layer probabilities from hidden layer.
        
        Parameters
        ----------
        hidden : torch.Tensor
            Hidden layer activations.
            
        Returns
        -------
        torch.Tensor
            Visible layer probabilities.
        """
        pass
    
    @abstractmethod
    def energy(self, visible: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute energy of visible-hidden configuration.
        
        Lower energy = higher probability configuration.
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible layer activations.
        hidden : torch.Tensor
            Hidden layer activations.
            
        Returns
        -------
        torch.Tensor
            Energy values (one per sample in batch).
        """
        pass
    
    def negative_energy(self, visible: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
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
    
    @classmethod
    def copy_rbm_weights(cls, source_rbm: 'RBM', target_rbm: 'RBM') -> None:
        """Copy weights and biases from source to target RBM.
        
        Parameters
        ----------
        source_rbm : RBM
            Source RBM to copy from.
        target_rbm : RBM
            Target RBM to copy to.
            
        Raises
        ------
        ValueError
            If weight shapes don't match.
        """
        if source_rbm.weights.shape != target_rbm.weights.shape:
            raise ValueError(
                "Weights shape of source RBM doesn't match target RBM. "
                "They may be different classes."
            )
        
        with torch.no_grad():
            target_rbm.weights.copy_(source_rbm.weights.clone().detach())
            target_rbm.hidden_bias.copy_(source_rbm.hidden_bias.clone().detach())
            target_rbm.visible_bias.copy_(source_rbm.visible_bias.clone().detach())
    
    @abstractmethod
    def sample_from_hidden_probs(self, hidden_probs: torch.Tensor) -> torch.Tensor:
        """Sample hidden states from probabilities.
        
        Parameters
        ----------
        hidden_probs : torch.Tensor
            Hidden layer probabilities.
            
        Returns
        -------
        torch.Tensor
            Sampled hidden states.
        """
        pass
    
    @abstractmethod
    def sample_from_visible_probs(self, visible_probs: torch.Tensor) -> torch.Tensor:
        """Sample visible states from probabilities.
        
        Parameters
        ----------
        visible_probs : torch.Tensor
            Visible layer probabilities.
            
        Returns
        -------
        torch.Tensor
            Sampled visible states.
        """
        pass
    
    def contrastive_divergence(
        self, input_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform Contrastive Divergence (CD) algorithm.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input visible data.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (positive_visible, positive_hidden, negative_visible, negative_hidden)
        """
        with torch.no_grad():
            positive_hidden_probs = self.calc_hidden_probs(input_data)
            positive_hidden = self.sample_from_hidden_probs(positive_hidden_probs)
            
            hidden = positive_hidden
            for _ in range(self.cd_k):
                visible_probs = self.calc_visible_probs(hidden)
                visible = self.sample_from_visible_probs(visible_probs)
                
                hidden_probs = self.calc_hidden_probs(visible)
                hidden = self.sample_from_hidden_probs(hidden_probs)
        
        return input_data, positive_hidden, visible, hidden
    
    @abstractmethod
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Predict output labels for input batch.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch of visible data.
            
        Returns
        -------
        torch.Tensor
            Predicted labels.
        """
        pass
    
    def print_energy_metrics(self, input_data: torch.Tensor) -> float:
        """Calculate and print energy metrics for monitoring.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input data batch.
            
        Returns
        -------
        float
            Sum of energy values.
        """
        h = self.calc_hidden_probs(input_data)
        energy = self.energy(input_data, h).detach().cpu().numpy()
        energy_sum = np.sum(energy)
        self.energies_sum.append(energy_sum)
        print(f'Energy - sum:{energy_sum}, mean:{np.mean(energy)}')
        return energy_sum


def one_hot_encode(input_tensor: torch.Tensor, num_classes: int, relaxed: bool = False) -> torch.Tensor:
    """One-hot encode integer tensor.
    
    Parameters
    ----------
    input_tensor : torch.Tensor
        Integer tensor to encode.
    num_classes : int
        Number of classes (size of one-hot dimension).
    relaxed : bool, optional
        If True, invalid values (-1 or >= num_classes) become all-zeros.
        If False, raises error on invalid values. Default is False.
        
    Returns
    -------
    torch.Tensor
        One-hot encoded tensor.
        
    Raises
    ------
    ValueError
        If relaxed=False and input contains invalid values.
    """
    tensor = input_tensor.clone()
    
    if len(tensor.shape) < 1:
        tensor = tensor.unsqueeze(0)
    
    tensor = tensor.long()
    
    if relaxed:
        # Map invalid values to a fictional label that will be stripped
        fictional_label = num_classes
        invalid_mask = (tensor < 0) | (tensor >= num_classes)
        tensor[invalid_mask] = fictional_label
    else:
        if torch.any(tensor < 0) or torch.any(tensor >= num_classes):
            raise ValueError("Input values must be between 0 and num_classes - 1")
    
    one_hot = F.one_hot(tensor, num_classes + 1)
    
    # Strip the fictional label dimension
    return one_hot[..., :-1]


class MultinomialRBM(RBM):
    """Multinomial RBM with multi-class visible and hidden units.
    
    This class extends RBM with multinomial-specific functionality:
    - Multi-class visible units (one per classifier)
    - Multi-class hidden units (learned representation)
    - Preprocess method for hard/soft label handling
    
    Parameters
    ----------
    dx : int
        Number of visible units (classifiers).
    dh : int
        Number of hidden units.
    k : int
        Number of classes for the classification task.
    l : int, optional
        Number of multinomial visible states. Default is k.
    m : int, optional
        Number of multinomial hidden states. Default is k.
    device : torch.device or str
        Device for computations.
    temperature : float, optional
        Temperature for softmax. Default is 1.0.
    init_method : str, optional
        Weight initialization: 'rand', 'mv', 'mv_rand', etc. Default is 'rand'.
    randn_impact : float, optional
        Scale for random initialization. Default is 0.01.
    irbm_mv_impact : float, optional
        Scale for majority vote initialization. Default is 2.0.
        
    Attributes
    ----------
    weights : nn.Parameter
        Weight tensor of shape (l, m, dx, dh).
    visible_bias : nn.Parameter
        Visible bias of shape (dx, l).
    hidden_bias : nn.Parameter
        Hidden bias of shape (dh, m).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.k = kwargs.get('k')
        self.l = kwargs.get('l', self.k)
        self.m = kwargs.get('m', self.k)
        self.temperature = kwargs.get('temperature', 1.0)
        
        self.init_method = kwargs.get('init_method', 'rand')
        self.randn_impact = kwargs.get('randn_impact', 0.01)
        self.irbm_mv_impact = kwargs.get('irbm_mv_impact', 2.0)
        
        # Initialize weights
        self._init_weights(self.init_method)
        
        # Softmax layers
        self.softmax_hidden = nn.Softmax(dim=2)
        self.softmax_visible = nn.Softmax(dim=1)
    
    def _init_weights(self, init_method: str) -> None:
        """Initialize weights, visible bias, and hidden bias.
        
        Parameters
        ----------
        init_method : str
            Initialization method: 'rand', 'mv', 'mv_rand', 'mv_linear', etc.
        """
        if init_method.startswith('mv'):
            if init_method == 'mv_rand':
                weights = torch.randn(self.l, self.m, self.dx, self.dh) * self.randn_impact
                visible_bias = torch.randn(self.dx, self.l) * self.randn_impact
                hidden_bias = torch.randn(self.dh, self.m) * self.randn_impact
            elif init_method == 'mv_lo':
                weights = (torch.randn(self.l, self.m, self.dx, self.dh) * 
                          torch.randn(self.l, self.m, self.dx, self.dh) * self.randn_impact)
                visible_bias = (torch.randn(self.dx, self.l) * 
                               torch.randn(self.dx, self.l) * self.randn_impact)
                hidden_bias = (torch.randn(self.dh, self.m) * 
                              torch.randn(self.dh, self.m) * self.randn_impact)
            else:  # 'mv' base case
                weights = torch.zeros(self.l, self.m, self.dx, self.dh)
                visible_bias = torch.zeros(self.dx, self.l)
                hidden_bias = torch.zeros(self.dh, self.m)
            
            # Set diagonal weights for majority vote initialization
            for i in range(min(self.l, self.m)):
                weights[i, i] = torch.ones(self.dx, self.dh) * self.irbm_mv_impact
        else:  # 'rand' default
            weights = torch.randn(self.l, self.m, self.dx, self.dh) * self.randn_impact
            visible_bias = torch.randn(self.dx, self.l) * self.randn_impact
            hidden_bias = torch.randn(self.dh, self.m) * self.randn_impact
        
        # Register as parameters
        self.weights = nn.Parameter(weights.to(self.device))
        self.visible_bias = nn.Parameter(visible_bias.to(self.device))
        self.hidden_bias = nn.Parameter(hidden_bias.to(self.device))
    
    def save_fixed_weights(self) -> None:
        """Save fixed weights for the first hidden unit (dh==1).
        
        When dh=1, the first slice of weights and biases are treated as
        structural anchors that should not be updated during training.
        This method saves these initial values.
        
        This mechanism is part of the validated production architecture.
        It provides a stable reference frame for energy calculations and
        prevents certain pathological modes during training.
        """
        if self.dh == 1:
            with torch.no_grad():
                self.weights_l = self.weights[0].clone().detach().to(self.device)
                self.weights_m = self.weights.permute(1, 3, 0, 2)[0].clone().detach().to(self.device)
                self.hidden_mask = self.hidden_bias[:, 0].clone().detach().to(self.device)
                self.visible_mask = self.visible_bias[:, 0].clone().detach().to(self.device)
    
    def refreeze_weights(self) -> None:
        """Restore fixed weights from saved values.
        
        This method can be used to reset the fixed weights to their
        saved values if they were accidentally modified.
        
        Only applies when dh=1 (single hidden unit case).
        """
        if self.dh == 1:
            with torch.no_grad():
                self.weights[0] = self.weights_l.clone().detach().to(self.device)
                self.weights.permute(1, 3, 0, 2)[0] = self.weights_m.clone().detach().to(self.device)
                self.hidden_bias[:, 0] = self.hidden_mask.clone().detach().to(self.device)
                self.visible_bias[:, 0] = self.visible_mask.clone().detach().to(self.device)
    
    def print_fixed_weights_sum(self) -> float:
        """Print sum of fixed weights for debugging/monitoring.
        
        This should remain constant during training if weight freezing
        is working correctly.
        
        Returns
        -------
        float
            Sum of all fixed weights, or 0.0 if dh != 1.
        """
        if self.dh == 1:
            sum_val = 0.0
            sum_val += self.weights[0].detach().cpu().sum().item()
            sum_val += self.weights.permute(1, 3, 0, 2)[0].detach().cpu().sum().item()
            sum_val += self.hidden_bias[:, 0].detach().cpu().sum().item()
            sum_val += self.visible_bias[:, 0].detach().cpu().sum().item()
            # Uncomment for debugging:
            # print(f'Fixed weights sum: {sum_val}')
            return sum_val
        return 0.0
    
    def zero_fixed_weights_grad(self) -> None:
        """Zero gradients of fixed weights to prevent updates.
        
        CRITICAL: This must be called after loss.backward() and before
        optimizer.step() to prevent the "fixed" weights from being updated.
        
        This only applies when dh=1 (single hidden unit case).
        When dh > 1, this is a no-op.
        
        The trainer (RBMTrainer) automatically calls this method if it exists.
        """
        if self.dh == 1 and self.weights.grad is not None:
            with torch.no_grad():
                self.weights.grad[0].zero_()
                self.weights.grad.permute(1, 3, 0, 2)[0].zero_()
                self.visible_bias.grad[:, 0].zero_()
                self.hidden_bias.grad[:, 0].zero_()
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Convert hard labels to one-hot or pass through soft labels.
        
        This is the critical method for handling both data formats:
        - Hard labels (N, D): Integer predictions from classifiers
        - Soft labels (N, K, D): Probability distributions
        
        Parameters
        ----------
        data : torch.Tensor
            Either (N, D) hard labels or (N, K, D) soft labels.
            Hard labels can contain -1 for missing values.
            
        Returns
        -------
        torch.Tensor
            (N, K, D) one-hot or probability tensor.
            Missing values (-1) become all-zero vectors.
        """
        if len(data.shape) < 3:
            # Hard labels: convert to one-hot
            with torch.no_grad():
                x_int = data.long()
                # Use relaxed mode to handle -1 (missing values) gracefully
                x_oh = one_hot_encode(x_int, self.l, relaxed=True)
                # Permute from (N, D, K) to (N, K, D)
                x_oh = x_oh.permute(0, 2, 1).float()
            return x_oh
        # Soft labels: pass through unchanged
        return data
    
    def calc_hidden_probs(self, visible: torch.Tensor) -> torch.Tensor:
        """Calculate hidden probabilities from visible layer.
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible activations of shape (N, L, D).
            
        Returns
        -------
        torch.Tensor
            Hidden probabilities of shape (N, H, M).
        """
        hidden_logits = torch.einsum('lmij,nli->njm', self.weights, visible) + self.hidden_bias
        hidden_probs = self.softmax_hidden(hidden_logits / self.temperature)
        return hidden_probs
    
    def calc_visible_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """Calculate visible probabilities from hidden layer.
        
        Parameters
        ----------
        hidden : torch.Tensor
            Hidden activations of shape (N, H, M).
            
        Returns
        -------
        torch.Tensor
            Visible probabilities of shape (N, L, D).
        """
        visible_logits = torch.einsum('lmij,njm->nli', self.weights, hidden) + self.visible_bias.T
        visible_probs = self.softmax_visible(visible_logits / self.temperature)
        return visible_probs
    
    def sample_from_hidden_probs(self, hidden_probs: torch.Tensor) -> torch.Tensor:
        """Sample from hidden probabilities.
        
        Parameters
        ----------
        hidden_probs : torch.Tensor
            Hidden probabilities of shape (N, H, M).
            
        Returns
        -------
        torch.Tensor
            Sampled hidden states (one-hot) of shape (N, H, M).
        """
        if self.deterministic:
            return hidden_probs
        
        with torch.no_grad():
            if self.as_distribution:
                samples = Categorical(hidden_probs).sample()
            else:
                samples = hidden_probs.argmax(dim=2)
            return F.one_hot(samples, self.m).float()
    
    def sample_from_visible_probs(self, visible_probs: torch.Tensor) -> torch.Tensor:
        """Sample from visible probabilities.
        
        Parameters
        ----------
        visible_probs : torch.Tensor
            Visible probabilities of shape (N, L, D).
            
        Returns
        -------
        torch.Tensor
            Sampled visible states (one-hot) of shape (N, L, D).
        """
        if self.deterministic:
            return visible_probs
        
        with torch.no_grad():
            if self.as_distribution:
                # Permute to (N, D, L) for Categorical, then back
                samples = Categorical(visible_probs.permute(0, 2, 1)).sample()
            else:
                samples = visible_probs.argmax(dim=1)
            return F.one_hot(samples, self.l).float().permute(0, 2, 1)
    
    def contrastive_divergence(
        self, input_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform CD with preprocessing.
        
        Parameters
        ----------
        input_data : torch.Tensor
            Input data (hard or soft labels).
            
        Returns
        -------
        Tuple
            (positive_visible, positive_hidden, negative_visible, negative_hidden)
        """
        input_data = self.preprocess(input_data)
        return super().contrastive_divergence(input_data)
    
    def energy(self, visible: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute energy of visible-hidden configuration.
        
        Parameters
        ----------
        visible : torch.Tensor
            Visible activations of shape (N, L, D).
        hidden : torch.Tensor
            Hidden activations of shape (N, H, M).
            
        Returns
        -------
        torch.Tensor
            Energy values of shape (N,).
        """
        v_flat = visible.flatten(1)  # (N, L*D)
        
        visible_term = torch.matmul(v_flat, self.visible_bias.T.flatten())
        hidden_term = torch.matmul(hidden.flatten(1), self.hidden_bias.flatten())
        weight_term = torch.einsum('lmij,nli,njm->n', self.weights, visible, hidden)
        
        return -(weight_term + hidden_term + visible_term)
    
    def predict(self, batch: torch.Tensor, as_distribution: bool = False) -> torch.Tensor:
        """Predict labels for input batch.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch (hard or soft labels).
        as_distribution : bool, optional
            If True, sample from distribution; if False, use argmax.
            Default is False.
            
        Returns
        -------
        torch.Tensor
            Predicted labels of shape (N, H).
        """
        batch = self.preprocess(batch)
        batch_probs = self.calc_hidden_probs(batch)
        
        if as_distribution:
            return Categorical(batch_probs).sample()
        return batch_probs.argmax(dim=2)
    
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
        return super().print_energy_metrics(input_data)

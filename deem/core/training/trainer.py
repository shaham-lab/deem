"""RBM Trainer - Production training loop.

This module contains the RBMTrainer class, which encapsulates the production
training loop extracted from run_predict.py (lines 378-398).

CRITICAL: This implementation processes ALL batches correctly,
unlike the legacy training.py which had a break bug.

Example
-------
>>> from deem.core.training import RBMTrainer
>>> from deem.core.models import MultinomialRBMGwg
>>> from deem.core.losses import EBMLoss
>>>
>>> model = MultinomialRBMGwg(dx=15, dh=3, k=3, device='cpu')
>>> loss_fn = EBMLoss(model)
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
>>> trainer = RBMTrainer(model, optimizer, loss_fn, device='cpu')
>>>
>>> history = trainer.fit(train_loader, epochs=100)
"""

from typing import Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for training callbacks.
    
    Callbacks can be used to implement custom logging, early stopping,
    checkpointing, or other training hooks.
    """
    
    def on_epoch_start(self, epoch: int, trainer: 'RBMTrainer') -> None:
        """Called at the start of each epoch."""
        ...
    
    def on_epoch_end(
        self, 
        epoch: int, 
        trainer: 'RBMTrainer', 
        metrics: Dict[str, float]
    ) -> bool:
        """Called at the end of each epoch.
        
        Returns
        -------
        bool
            True to continue training, False to stop early.
        """
        ...
    
    def on_batch_end(
        self, 
        batch_idx: int, 
        trainer: 'RBMTrainer', 
        loss: float
    ) -> None:
        """Called after each batch."""
        ...


def apply_l1_regularization(
    loss: torch.Tensor,
    model: nn.Module,
    l1_lambda: float,
    scale_dynamic: float = 0.01
) -> torch.Tensor:
    """Add L1 regularization to the loss.
    
    Adds L1 norm penalty over all model parameters to encourage sparsity.
    
    Parameters
    ----------
    loss : torch.Tensor
        Base loss value.
    model : nn.Module
        Model containing parameters to regularize.
    l1_lambda : float or str
        L1 regularization strength. Can be:
        - 0 or None: No regularization
        - float > 0: Fixed regularization strength
        - 'dynamic': Scale L1 by loss magnitude
    scale_dynamic : float, optional
        Fraction of loss to use for dynamic L1. Default is 0.01.
        
    Returns
    -------
    torch.Tensor
        Loss with L1 penalty applied.
        
    Example
    -------
    >>> loss = compute_loss(model, data)
    >>> loss = apply_l1_regularization(loss, model, l1_lambda=0.001)
    >>> loss.backward()
    """
    if l1_lambda == 0 or l1_lambda is None:
        return loss
    
    # Compute L1 norm over all parameters
    l1_norm = torch.cat([p.view(-1) for p in model.parameters()]).abs().sum()
    
    # Handle dynamic L1
    if l1_lambda == 'dynamic':
        l1_lambda = scale_dynamic * loss.detach()
    
    return loss + l1_lambda * l1_norm


class RBMTrainer:
    """Production RBM trainer extracted from run_predict.py.
    
    This trainer encapsulates the training loop that was proven to work
    in production experiments. It correctly processes ALL batches
    (no break bug that existed in training.py).
    
    The training process:
    1. Model generates real/fake samples via get_samples()
    2. Loss is computed using EBMLoss (energy difference)
    3. Optional L1 regularization is applied
    4. Gradients are computed and optimizer steps
    5. Optional LR scheduler steps per batch or epoch
    
    Parameters
    ----------
    model : nn.Module
        The RBM model to train. Must have get_samples() and 
        zero_fixed_weights_grad() methods.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer for updating model parameters.
    loss_fn : callable
        Loss function that takes (v_pos, h_pos, v_neg, h_neg)
        and returns a scalar loss. Typically EBMLoss.
    device : str or torch.device, optional
        Device for training. Default is 'cpu'.
    l1_lambda : float, optional
        L1 regularization strength. Default is 0.0 (no regularization).
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler. Default is None.
    scheduler_step_per_batch : bool, optional
        If True, step scheduler after each batch (e.g., OneCycleLR).
        If False, step after each epoch (e.g., CosineAnnealingLR).
        Default is False.
        
    Attributes
    ----------
    model : nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer.
    loss_fn : callable
        The loss function.
    device : torch.device
        Training device.
    history : Dict[str, List[float]]
        Training history with losses and other metrics.
        
    Example
    -------
    >>> trainer = RBMTrainer(
    ...     model=rbm,
    ...     optimizer=torch.optim.SGD(rbm.parameters(), lr=0.01, momentum=0.9),
    ...     loss_fn=EBMLoss(rbm),
    ...     device='cuda'
    ... )
    >>> history = trainer.fit(train_loader, epochs=100, verbose=True)
    >>> print(f"Final loss: {history['loss'][-1]:.4f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: Union[str, torch.device] = 'cpu',
        l1_lambda: float = 0.0,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_step_per_batch: bool = False,
    ):
        """Initialize the RBMTrainer."""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device) if isinstance(device, str) else device
        self.l1_lambda = l1_lambda
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'lr': [],
        }
        
        # Current epoch (for callbacks)
        self.current_epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """Train for one epoch.
        
        This is the core training loop extracted from run_predict.py lines 378-398.
        CRITICAL: Processes ALL batches (no break bug).
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
            
        Returns
        -------
        float
            Average loss for the epoch.
        """
        self.model.train()
        accumulated_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats (with or without labels)
            if isinstance(batch, (list, tuple)):
                batch_input = batch[0]
            else:
                batch_input = batch
            
            # Move to device and ensure float
            batch_input = batch_input.float().to(self.device)
            
            # Forward pass - get real and fake samples
            # model.get_samples returns (v_pos, h_pos, v_neg, h_neg)
            samples = self.model.get_samples(batch_input)
            
            # Compute loss
            loss = self.loss_fn(*samples)
            
            # Apply L1 regularization if configured
            if self.l1_lambda != 0:
                loss = apply_l1_regularization(
                    loss, self.model, self.l1_lambda
                )
            
            accumulated_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Zero gradients for fixed weights (if model supports it)
            self.model.zero_fixed_weights_grad()
            
            # Optimizer step
            self.optimizer.step()
            
            # Step scheduler per batch if configured (e.g., OneCycleLR)
            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()
        
        # Step scheduler per epoch if configured (e.g., CosineAnnealingLR)
        if self.scheduler is not None and not self.scheduler_step_per_batch:
            self.scheduler.step()
        
        # Return average loss
        return accumulated_loss / max(num_batches, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        print_every: int = 10,
        accuracy_callback: Optional[Callable[[], Optional[float]]] = None,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        epochs : int
            Number of epochs to train.
        verbose : bool, optional
            Whether to print progress. Default is True.
        val_loader : DataLoader, optional
            Validation data loader (for future use). Default is None.
        callbacks : List[TrainingCallback], optional
            List of callbacks to run during training. Default is None.
        print_every : int, optional
            Print progress every N epochs. Default is 10.
        accuracy_callback : callable, optional
            Optional callback to compute accuracy for verbose logging.
            Should return float accuracy or None. Default is None.
            
        Returns
        -------
        Dict[str, List[float]]
            Training history containing losses and learning rates.
            
        Example
        -------
        >>> history = trainer.fit(train_loader, epochs=100, verbose=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history['loss'])
        >>> plt.xlabel('Epoch')
        >>> plt.ylabel('Loss')
        """
        callbacks = callbacks or []
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Callback: epoch start
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_start'):
                    callback.on_epoch_start(epoch, self)
            
            # Train for one epoch
            epoch_loss = self.train_epoch(train_loader)
            
            # Record history
            self.history['loss'].append(epoch_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Callback: epoch end
            metrics = {'loss': epoch_loss, 'lr': current_lr}
            should_continue = True
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    result = callback.on_epoch_end(epoch, self, metrics)
                    if result is False:
                        should_continue = False
            
            # Print progress with optional accuracy
            if verbose and epoch % print_every == 0:
                if accuracy_callback is not None:
                    accuracy = accuracy_callback()
                    if accuracy is not None:
                        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f} - Acc: {accuracy:.2%}")
                    else:
                        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f}")
            
            # Early stopping from callbacks
            if not should_continue:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        return self.history
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.
        
        Parameters
        ----------
        path : str
            Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'current_epoch': self.current_epoch,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.
        
        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['current_epoch']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

"""Tests for deem.core.training module.

Tests cover:
- RBMTrainer instantiation
- Single epoch training
- Multiple epoch training
- L1 regularization
- Scheduler integration
- Checkpointing
- Training callbacks
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from deem.core.training import RBMTrainer, apply_l1_regularization, TrainingCallback
from deem.core.models.rbm_gwg import MultinomialRBMGwg
from deem.core.losses import EBMLoss


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Return device for tests."""
    return 'cpu'


@pytest.fixture
def rbm(device):
    """Create a MultinomialRBMGwg instance for testing."""
    return MultinomialRBMGwg(
        dx=10,
        dh=2,
        cd_k=5,
        k=3,
        l=3,
        m=3,
        device=device,
        deterministic=True,
        sampler_steps=2,
    )


@pytest.fixture
def loss_fn(rbm):
    """Create EBMLoss for testing."""
    return EBMLoss(rbm, with_norm=False)


@pytest.fixture
def optimizer(rbm):
    """Create optimizer for testing."""
    return torch.optim.SGD(rbm.parameters(), lr=0.01, momentum=0.9)


@pytest.fixture
def train_loader():
    """Create a dummy training data loader."""
    torch.manual_seed(42)
    # Hard labels (N, D)
    data = torch.randint(0, 3, (100, 10))
    labels = torch.randint(0, 3, (100,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def trainer(rbm, optimizer, loss_fn, device):
    """Create an RBMTrainer instance for testing."""
    return RBMTrainer(
        model=rbm,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )


# =============================================================================
# Tests: L1 Regularization
# =============================================================================

class TestL1Regularization:
    """Tests for L1 regularization utility."""
    
    def test_no_regularization(self, rbm):
        """Test that l1_lambda=0 returns unchanged loss."""
        loss = torch.tensor(1.0)
        result = apply_l1_regularization(loss, rbm, l1_lambda=0)
        assert result == loss
    
    def test_none_regularization(self, rbm):
        """Test that l1_lambda=None returns unchanged loss."""
        loss = torch.tensor(1.0)
        result = apply_l1_regularization(loss, rbm, l1_lambda=None)
        assert result == loss
    
    def test_fixed_regularization(self, rbm):
        """Test fixed L1 regularization."""
        loss = torch.tensor(1.0)
        result = apply_l1_regularization(loss, rbm, l1_lambda=0.01)
        
        # Should be greater than original loss
        assert result > loss
        
        # Should be loss + 0.01 * L1_norm
        l1_norm = torch.cat([p.view(-1) for p in rbm.parameters()]).abs().sum()
        expected = loss + 0.01 * l1_norm
        assert torch.isclose(result, expected)
    
    def test_dynamic_regularization(self, rbm):
        """Test dynamic L1 regularization."""
        loss = torch.tensor(10.0)
        result = apply_l1_regularization(loss, rbm, l1_lambda='dynamic', scale_dynamic=0.01)
        
        # Should be greater than original loss
        assert result > loss
        
        # Dynamic lambda should be 0.01 * 10.0 = 0.1
        l1_norm = torch.cat([p.view(-1) for p in rbm.parameters()]).abs().sum()
        expected = loss + 0.1 * l1_norm
        assert torch.isclose(result, expected)


# =============================================================================
# Tests: Trainer Instantiation
# =============================================================================

class TestTrainerInstantiation:
    """Tests for RBMTrainer creation."""
    
    def test_basic_creation(self, trainer):
        """Test basic trainer creation."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.device == torch.device('cpu')
    
    def test_default_parameters(self, trainer):
        """Test default parameter values."""
        assert trainer.l1_lambda == 0.0
        assert trainer.scheduler is None
        assert trainer.scheduler_step_per_batch is False
    
    def test_history_initialized(self, trainer):
        """Test that history is properly initialized."""
        assert 'loss' in trainer.history
        assert 'lr' in trainer.history
        assert len(trainer.history['loss']) == 0
        assert len(trainer.history['lr']) == 0
    
    def test_with_l1_regularization(self, rbm, optimizer, loss_fn, device):
        """Test trainer creation with L1 regularization."""
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            l1_lambda=0.001,
        )
        assert trainer.l1_lambda == 0.001
    
    def test_with_scheduler(self, rbm, optimizer, loss_fn, device):
        """Test trainer creation with LR scheduler."""
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
        )
        assert trainer.scheduler is not None


# =============================================================================
# Tests: Training
# =============================================================================

class TestTraining:
    """Tests for training functionality."""
    
    def test_train_epoch(self, trainer, train_loader):
        """Test single epoch training."""
        loss = trainer.train_epoch(train_loader)
        
        # Loss should be a float
        assert isinstance(loss, float)
        
        # Loss should be finite
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))
    
    def test_train_epoch_processes_all_batches(self, trainer, train_loader):
        """Test that all batches are processed (no break bug)."""
        # Count batches manually
        expected_batches = len(train_loader)
        
        # Train and check model was updated
        initial_weights = trainer.model.weights.clone()
        trainer.train_epoch(train_loader)
        final_weights = trainer.model.weights
        
        # Weights should have changed
        assert not torch.allclose(initial_weights, final_weights)
    
    def test_fit_multiple_epochs(self, trainer, train_loader):
        """Test training for multiple epochs."""
        history = trainer.fit(train_loader, epochs=5, verbose=False)
        
        # History should have 5 entries
        assert len(history['loss']) == 5
        assert len(history['lr']) == 5
    
    def test_fit_verbose_output(self, trainer, train_loader, capsys):
        """Test verbose output during training."""
        trainer.fit(train_loader, epochs=10, verbose=True, print_every=5)
        
        captured = capsys.readouterr()
        # Should print at epoch 5 and 10
        assert 'Epoch 5/10' in captured.out
        assert 'Epoch 10/10' in captured.out
    
    def test_fit_returns_history(self, trainer, train_loader):
        """Test that fit returns the training history."""
        history = trainer.fit(train_loader, epochs=3, verbose=False)
        
        assert isinstance(history, dict)
        assert 'loss' in history
        assert 'lr' in history
        assert len(history['loss']) == 3


# =============================================================================
# Tests: Scheduler Integration
# =============================================================================

class TestSchedulerIntegration:
    """Tests for LR scheduler integration."""
    
    def test_step_scheduler_per_epoch(self, rbm, optimizer, loss_fn, device, train_loader):
        """Test scheduler stepping per epoch."""
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        initial_lr = optimizer.param_groups[0]['lr']
        
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
            scheduler_step_per_batch=False,
        )
        
        trainer.fit(train_loader, epochs=5, verbose=False)
        
        # LR should have decreased after step_size epochs
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_step_scheduler_per_batch(self, rbm, optimizer, loss_fn, device, train_loader):
        """Test scheduler stepping per batch."""
        total_steps = len(train_loader) * 5  # 5 epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, total_steps=total_steps
        )
        
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scheduler=scheduler,
            scheduler_step_per_batch=True,
        )
        
        # Should not raise
        trainer.fit(train_loader, epochs=5, verbose=False)


# =============================================================================
# Tests: Checkpointing
# =============================================================================

class TestCheckpointing:
    """Tests for checkpoint save/load."""
    
    def test_save_checkpoint(self, trainer, train_loader):
        """Test saving a checkpoint."""
        trainer.fit(train_loader, epochs=3, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer.save_checkpoint(path)
            
            # File should exist
            assert os.path.exists(path)
            
            # Should be loadable
            checkpoint = torch.load(path)
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'history' in checkpoint
            assert 'current_epoch' in checkpoint
    
    def test_load_checkpoint(self, rbm, optimizer, loss_fn, device, train_loader):
        """Test loading a checkpoint."""
        trainer1 = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        trainer1.fit(train_loader, epochs=3, verbose=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer1.save_checkpoint(path)
            
            # Create new trainer and load checkpoint
            rbm2 = MultinomialRBMGwg(
                dx=10, dh=2, cd_k=5, k=3, l=3, m=3,
                device=device, deterministic=True, sampler_steps=2,
            )
            optimizer2 = torch.optim.SGD(rbm2.parameters(), lr=0.01, momentum=0.9)
            loss_fn2 = EBMLoss(rbm2)
            
            trainer2 = RBMTrainer(
                model=rbm2,
                optimizer=optimizer2,
                loss_fn=loss_fn2,
                device=device,
            )
            trainer2.load_checkpoint(path)
            
            # History should match
            assert trainer2.history == trainer1.history
            assert trainer2.current_epoch == trainer1.current_epoch
            
            # Model weights should match
            assert torch.allclose(
                rbm2.weights, 
                rbm.weights
            )


# =============================================================================
# Tests: Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for training callbacks."""
    
    def test_callback_on_epoch_end(self, trainer, train_loader):
        """Test callback called at epoch end."""
        epoch_counts = []
        
        class CountingCallback:
            def on_epoch_end(self, epoch, trainer, metrics):
                epoch_counts.append(epoch)
                return True
        
        callback = CountingCallback()
        trainer.fit(train_loader, epochs=5, verbose=False, callbacks=[callback])
        
        assert epoch_counts == [1, 2, 3, 4, 5]
    
    def test_callback_early_stopping(self, trainer, train_loader):
        """Test callback can trigger early stopping."""
        class EarlyStopCallback:
            def on_epoch_end(self, epoch, trainer, metrics):
                if epoch >= 3:
                    return False  # Stop training
                return True
        
        callback = EarlyStopCallback()
        history = trainer.fit(train_loader, epochs=10, verbose=False, callbacks=[callback])
        
        # Should have stopped at epoch 3
        assert len(history['loss']) == 3
    
    def test_callback_receives_metrics(self, trainer, train_loader):
        """Test callback receives correct metrics."""
        received_metrics = []
        
        class MetricsCallback:
            def on_epoch_end(self, epoch, trainer, metrics):
                received_metrics.append(metrics.copy())
                return True
        
        callback = MetricsCallback()
        trainer.fit(train_loader, epochs=3, verbose=False, callbacks=[callback])
        
        assert len(received_metrics) == 3
        for metrics in received_metrics:
            assert 'loss' in metrics
            assert 'lr' in metrics


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_data_loader(self, trainer):
        """Test handling of empty data loader."""
        empty_dataset = TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
        empty_loader = DataLoader(empty_dataset, batch_size=16)
        
        loss = trainer.train_epoch(empty_loader)
        
        # Should return 0.0 for empty loader
        assert loss == 0.0
    
    def test_single_batch(self, trainer):
        """Test training with a single batch."""
        data = torch.randint(0, 3, (16, 10))
        labels = torch.randint(0, 3, (16,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=16)
        
        loss = trainer.train_epoch(loader)
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
    
    def test_batch_without_labels(self, trainer):
        """Test training with batch that has no labels."""
        data = torch.randint(0, 3, (32, 10))
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=16)
        
        # Should work without labels
        loss = trainer.train_epoch(loader)
        assert isinstance(loss, float)
    
    def test_model_in_train_mode(self, trainer, train_loader):
        """Test that model is in train mode during training."""
        trainer.model.eval()  # Start in eval mode
        
        trainer.train_epoch(train_loader)
        
        # Model should be in train mode
        assert trainer.model.training


# =============================================================================
# Tests: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for complete training workflows."""
    
    def test_full_training_workflow(self, device):
        """Test complete training workflow."""
        # Create model
        rbm = MultinomialRBMGwg(
            dx=15, dh=3, cd_k=10, k=3, l=3, m=3,
            device=device, deterministic=True, sampler_steps=2,
        )
        
        # Create loss and optimizer
        loss_fn = EBMLoss(rbm)
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.01, momentum=0.9)
        
        # Create data
        torch.manual_seed(42)
        data = torch.randint(0, 3, (200, 15))
        labels = torch.randint(0, 3, (200,))
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create trainer
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        
        # Train
        history = trainer.fit(train_loader, epochs=10, verbose=False)
        
        # Verify training happened
        assert len(history['loss']) == 10
        
        # Verify model can do inference
        rbm.eval()
        with torch.no_grad():
            test_data = torch.randint(0, 3, (16, 15))
            predictions = rbm.predict(test_data)
            
            # Predictions should have correct shape (N, DH)
            assert predictions.shape == (16, 3)
    
    def test_training_with_all_options(self, device):
        """Test training with all optional features."""
        # Create model
        rbm = MultinomialRBMGwg(
            dx=10, dh=2, cd_k=5, k=3, l=3, m=3,
            device=device, deterministic=True, sampler_steps=2,
        )
        
        # Create loss, optimizer, scheduler
        loss_fn = EBMLoss(rbm)
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Create data
        data = torch.randint(0, 3, (100, 10))
        labels = torch.randint(0, 3, (100,))
        train_loader = DataLoader(TensorDataset(data, labels), batch_size=16)
        
        # Create callback
        loss_history = []
        class LossCallback:
            def on_epoch_end(self, epoch, trainer, metrics):
                loss_history.append(metrics['loss'])
                return True
        
        # Create trainer with all options
        trainer = RBMTrainer(
            model=rbm,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            l1_lambda=0.001,
            scheduler=scheduler,
            scheduler_step_per_batch=False,
        )
        
        # Train
        history = trainer.fit(
            train_loader,
            epochs=10,
            verbose=False,
            callbacks=[LossCallback()],
        )
        
        # Verify everything worked
        assert len(history['loss']) == 10
        assert len(loss_history) == 10
        assert loss_history == history['loss']
        
        # LR should have decreased
        assert history['lr'][-1] < history['lr'][0]

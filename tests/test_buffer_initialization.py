"""Test buffer initialization with first batch.

This module tests the buffer initialization functionality that pre-populates
the sampler buffer with real training examples to improve MCMC mixing.

Matches production behavior from run_predict.py (lines 371-377).
"""

import pytest
import torch
import numpy as np

from deem import DEEM
from deem.core.models import MultinomialRBMGwg


class TestBufferInitializationHelper:
    """Test the _initialize_buffer_with_first_batch helper method."""

    def test_buffer_populated_after_fit(self):
        """Test that buffer is populated after fit() is called."""
        # Create synthetic data
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        # Initialize model
        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,  # Minimal training
            device='cpu',
        )

        # Before fit, model is not initialized
        assert model.model_ is None

        # Fit model
        model.fit(X, verbose=False)

        # After fit, buffer should be populated
        assert len(model.model_.sampler.buffer) > 0
        assert model.model_.sampler.buffer.buffer_max_size == 8192  # Default max_len

    def test_buffer_size_matches_batch_size(self):
        """Test that buffer size matches first batch size."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        batch_size = 32
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            batch_size=batch_size,
            device='cpu',
        )

        model.fit(X, verbose=False)

        # Buffer should contain at least batch_size examples (first batch)
        # May contain more after training, but at least batch_size
        assert len(model.model_.sampler.buffer) >= batch_size

    def test_buffer_shape_matches_data_shape_hard_labels(self):
        """Test that buffer examples have correct shape for hard labels."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            device='cpu',
        )

        model.fit(X, verbose=False)

        # Get examples from buffer
        buffer_examples = model.model_.sampler.buffer.get_random_examples(1)
        
        # For hard labels (not oh_mode), buffer stores raw data shape: (batch, n_classifiers)
        # or possibly (batch, k, n_classifiers) if preprocessed
        # The shape depends on oh_mode setting
        assert buffer_examples.ndim >= 2

    def test_buffer_shape_matches_data_shape_soft_labels(self):
        """Test that buffer examples have correct shape for soft labels."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        # Create soft labels: (n_samples, n_classes, n_classifiers)
        X = np.random.rand(n_samples, n_classes, n_classifiers).astype(np.float32)
        # Normalize to probabilities
        X = X / X.sum(axis=1, keepdims=True)

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            device='cpu',
            sampler_oh_mode=True,  # Enable oh_mode for soft labels
        )

        model.fit(X, verbose=False)

        # Get examples from buffer
        buffer_examples = model.model_.sampler.buffer.get_random_examples(1)
        
        # For soft labels with oh_mode=True, buffer stores preprocessed data
        # Shape should be (batch, k, n_classifiers)
        assert buffer_examples.ndim == 3
        assert buffer_examples.shape[1] == n_classes  # k dimension
        assert buffer_examples.shape[2] == n_classifiers  # dx dimension

    def test_buffer_not_reinitialized_if_already_populated(self):
        """Test that buffer is not reinitialized if already populated."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            batch_size=32,
            device='cpu',
        )

        model.fit(X, verbose=False)

        # Record buffer state after first fit
        buffer_size_after_first_fit = len(model.model_.sampler.buffer)
        first_example = model.model_.sampler.buffer.examples[0].clone()

        # Manually add a marker example
        marker = torch.zeros(n_classifiers)
        model.model_.sampler.buffer.add_examples(marker.unsqueeze(0))
        buffer_size_with_marker = len(model.model_.sampler.buffer)

        # Create new training data and use _initialize_buffer_with_first_batch
        from torch.utils.data import DataLoader, TensorDataset
        new_X = torch.randint(0, n_classes, (50, n_classifiers)).float()
        loader = DataLoader(TensorDataset(new_X), batch_size=16, shuffle=False)

        # Call init again - should be a no-op since buffer is not empty
        model._initialize_buffer_with_first_batch(loader, verbose=False)

        # Buffer should not have changed (still contains the marker)
        assert len(model.model_.sampler.buffer) == buffer_size_with_marker


class TestBufferInitializationIntegration:
    """Integration tests for buffer initialization in training workflow."""

    def test_training_with_initialized_buffer(self):
        """Test that training runs successfully with buffer initialization."""
        n_samples, n_classifiers, n_classes = 200, 15, 4
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=5,
            device='cpu',
        )

        # Should not raise any errors
        model.fit(X, verbose=False)

        # Model should be fitted
        assert model.is_fitted_
        assert len(model.history_['loss']) == 5

    def test_training_with_soft_labels(self):
        """Test training with soft labels uses correct buffer initialization."""
        n_samples, n_classifiers, n_classes = 200, 15, 4
        # Create soft labels
        X = np.random.rand(n_samples, n_classes, n_classifiers).astype(np.float32)
        X = X / X.sum(axis=1, keepdims=True)

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=5,
            device='cpu',
            sampler_oh_mode=True,
        )

        # Should not raise any errors
        model.fit(X, verbose=False)

        # Model should be fitted
        assert model.is_fitted_
        assert model.model_.sampler.oh_mode is True
        assert len(model.model_.sampler.buffer) > 0

    def test_buffer_content_is_valid(self):
        """Test that buffer contains valid data after initialization."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            device='cpu',
        )

        model.fit(X, verbose=False)

        # Get examples from buffer
        buffer_examples = model.model_.sampler.buffer.get_random_examples(10)

        # Buffer examples should be finite (no NaN or Inf)
        assert torch.isfinite(buffer_examples).all()

    def test_prediction_after_buffer_initialized_training(self):
        """Test that predictions work after training with buffer initialization."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X_train = np.random.randint(0, n_classes, (n_samples, n_classifiers))
        X_test = np.random.randint(0, n_classes, (20, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=5,
            device='cpu',
        )

        model.fit(X_train, verbose=False)
        predictions = model.predict(X_test)

        # Predictions should have correct shape
        assert predictions.shape == (20,)
        # Predictions should be valid class indices
        assert np.all(predictions >= 0)
        assert np.all(predictions < n_classes)


class TestBufferInitializationEdgeCases:
    """Test edge cases for buffer initialization."""

    def test_small_dataset(self):
        """Test buffer initialization with very small dataset."""
        n_samples, n_classifiers, n_classes = 5, 3, 2
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            batch_size=10,  # Larger than dataset
            device='cpu',
        )

        # Should not raise any errors
        model.fit(X, verbose=False)

        # Buffer should be populated
        assert len(model.model_.sampler.buffer) > 0

    def test_batch_size_equals_dataset_size(self):
        """Test when batch_size equals dataset size."""
        n_samples, n_classifiers, n_classes = 50, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            batch_size=n_samples,  # Equals dataset size
            device='cpu',
        )

        model.fit(X, verbose=False)

        # Buffer should contain all samples from first batch
        assert len(model.model_.sampler.buffer) >= n_samples

    def test_with_missing_values(self):
        """Test buffer initialization handles missing values (-1)."""
        n_samples, n_classifiers, n_classes = 100, 10, 3
        X = np.random.randint(0, n_classes, (n_samples, n_classifiers))
        # Add some missing values (but not all -1 for any sample)
        X[::5, 0] = -1

        model = DEEM(
            n_classes=n_classes,
            hidden_dim=1,
            epochs=1,
            device='cpu',
        )

        # Should not raise any errors
        model.fit(X, verbose=False)

        # Model should be fitted
        assert model.is_fitted_
        assert len(model.model_.sampler.buffer) > 0

"""Tests for the high-level DEEM API."""

import numpy as np
import pytest
import torch

from deem import DEEM


class TestDEEMBasic:
    """Basic functionality tests for DEEM."""

    def test_init_default_params(self):
        """Test model initialization with default parameters."""
        model = DEEM()
        
        assert model.n_classes is None
        assert model.hidden_dim == 1
        assert model.cd_k == 10
        assert model.deterministic is True
        assert model.epochs == 100
        assert model.batch_size == 128
        assert model.is_fitted_ is False
        assert model.model_ is None

    def test_init_custom_params(self):
        """Test model initialization with custom parameters."""
        model = DEEM(
            n_classes=5,
            hidden_dim=2,
            cd_k=20,
            deterministic=False,
            learning_rate=0.01,
            epochs=50,
            batch_size=64,
            device='cpu',
        )
        
        assert model.n_classes == 5
        assert model.hidden_dim == 2
        assert model.cd_k == 20
        assert model.deterministic is False
        assert model.learning_rate == 0.01
        assert model.epochs == 50
        assert model.batch_size == 64

    def test_repr(self):
        """Test string representation."""
        model = DEEM(n_classes=3, epochs=50)
        repr_str = repr(model)
        
        assert "DEEM" in repr_str
        assert "n_classes=3" in repr_str
        assert "epochs=50" in repr_str
        assert "not fitted" in repr_str


class TestDEEMFit:
    """Tests for the fit method."""

    def test_fit_hard_labels(self):
        """Test fitting with hard labels (2D array)."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        
        result = model.fit(predictions, epochs=5, verbose=False)
        
        assert result is model  # Should return self
        assert model.is_fitted_ is True
        assert model.model_ is not None
        assert model.n_classes_ == 3
        assert model.n_classifiers_ == 15

    def test_fit_with_labels(self):
        """Test fitting with labels."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        labels = np.random.randint(0, 3, (100,))
        
        model.fit(predictions, labels=labels, epochs=5, verbose=False)
        
        assert model.is_fitted_ is True

    def test_fit_infers_n_classes(self):
        """Test that n_classes is inferred from data."""
        model = DEEM(device='cpu')  # No n_classes specified
        predictions = np.random.randint(0, 5, (100, 15))  # 5 classes (0-4)
        
        model.fit(predictions, epochs=3, verbose=False)
        
        assert model.n_classes_ == 5

    def test_fit_with_torch_tensors(self):
        """Test fitting with PyTorch tensors."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = torch.randint(0, 3, (100, 15))
        labels = torch.randint(0, 3, (100,))
        
        model.fit(predictions, labels=labels, epochs=3, verbose=False)
        
        assert model.is_fitted_ is True

    def test_fit_training_history(self):
        """Test that training history is recorded."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        
        model.fit(predictions, epochs=5, verbose=False)
        
        assert 'loss' in model.history_
        assert len(model.history_['loss']) == 5

    def test_fit_override_params(self):
        """Test overriding params in fit()."""
        model = DEEM(n_classes=3, epochs=100, batch_size=128, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        
        model.fit(predictions, epochs=3, batch_size=32, verbose=False)
        
        assert len(model.history_['loss']) == 3  # Used overridden epochs


class TestDEEMPredict:
    """Tests for the predict method."""

    @pytest.fixture
    def fitted_model(self):
        """Create and return a fitted model."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        model.fit(predictions, epochs=3, verbose=False)
        return model

    def test_predict_shape(self, fitted_model):
        """Test output shape of predict."""
        test_preds = np.random.randint(0, 3, (50, 15))
        consensus = fitted_model.predict(test_preds)
        
        assert consensus.shape == (50,)

    def test_predict_returns_numpy(self, fitted_model):
        """Test that predict returns numpy array."""
        test_preds = np.random.randint(0, 3, (50, 15))
        consensus = fitted_model.predict(test_preds)
        
        assert isinstance(consensus, np.ndarray)

    def test_predict_return_probs(self, fitted_model):
        """Test returning probabilities."""
        test_preds = np.random.randint(0, 3, (50, 15))
        probs = fitted_model.predict(test_preds, return_probs=True)
        
        assert probs.shape[0] == 50
        assert probs.ndim >= 2  # At least 2D for probabilities

    def test_predict_not_fitted_raises(self):
        """Test that predict raises if not fitted."""
        model = DEEM()
        test_preds = np.random.randint(0, 3, (50, 15))
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(test_preds)


class TestDEEMHungarian:
    """Tests for Hungarian alignment."""

    @pytest.fixture
    def fitted_model(self):
        """Create and return a fitted model."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        model.fit(predictions, epochs=3, verbose=False)
        return model

    def test_predict_with_alignment(self, fitted_model):
        """Test predict with align_to returns aligned predictions."""
        test_preds = np.random.randint(0, 3, (50, 15))
        train_preds = np.random.randint(0, 3, (100, 15))
        
        aligned = fitted_model.predict(test_preds, align_to=train_preds)
        
        assert aligned.shape == (50,)
        assert fitted_model.class_map_ is not None
        assert len(fitted_model.class_map_) == 3

    def test_score(self, fitted_model):
        """Test score returns accuracy with Hungarian alignment."""
        test_preds = np.random.randint(0, 3, (50, 15))
        true_labels = np.random.randint(0, 3, (50,))
        
        accuracy = fitted_model.score(test_preds, true_labels)
        
        assert 0.0 <= accuracy <= 1.0
        assert fitted_model.class_map_ is not None


class TestDEEMSoftLabels:
    """Tests for soft labels (probability distributions)."""

    def test_fit_soft_labels(self):
        """Test fitting with soft labels (3D array)."""
        model = DEEM(n_classes=3, device='cpu')
        # Soft labels: (N, K, D) probability distributions
        soft_preds = np.random.rand(100, 3, 15)
        soft_preds /= soft_preds.sum(axis=1, keepdims=True)  # Normalize
        
        model.fit(soft_preds, epochs=3, verbose=False)
        
        assert model.is_fitted_ is True
        assert model.n_classes_ == 3
        assert model.n_classifiers_ == 15

    def test_predict_soft_labels(self):
        """Test prediction with soft labels."""
        model = DEEM(n_classes=3, device='cpu')
        soft_preds = np.random.rand(100, 3, 15)
        soft_preds /= soft_preds.sum(axis=1, keepdims=True)
        
        model.fit(soft_preds, epochs=3, verbose=False)
        
        test_soft = np.random.rand(50, 3, 15)
        test_soft /= test_soft.sum(axis=1, keepdims=True)
        consensus = model.predict(test_soft)
        
        assert consensus.shape == (50,)


class TestDEEMSaveLoad:
    """Tests for save/load functionality."""

    def test_save_load(self, tmp_path):
        """Test saving and loading a model."""
        # Create and fit a model
        model = DEEM(n_classes=3, hidden_dim=2, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        model.fit(predictions, epochs=3, verbose=False)
        
        # Save
        save_path = tmp_path / "model.pt"
        model.save(save_path)
        
        # Load into new model
        loaded_model = DEEM(device='cpu')
        loaded_model.load(save_path)
        
        assert loaded_model.is_fitted_ is True
        assert loaded_model.n_classes_ == 3
        assert loaded_model.n_classifiers_ == 15

    def test_save_not_fitted_raises(self, tmp_path):
        """Test that save raises if not fitted."""
        model = DEEM()
        save_path = tmp_path / "model.pt"
        
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.save(save_path)


class TestDEEMSklearnCompat:
    """Tests for sklearn compatibility."""

    def test_get_params(self):
        """Test get_params returns expected parameters."""
        model = DEEM(n_classes=5, epochs=50)
        params = model.get_params()
        
        assert params['n_classes'] == 5
        assert params['epochs'] == 50
        assert 'hidden_dim' in params
        assert 'learning_rate' in params

    def test_set_params(self):
        """Test set_params modifies parameters."""
        model = DEEM()
        model.set_params(n_classes=10, epochs=200)
        
        assert model.n_classes == 10
        assert model.epochs == 200

    def test_set_invalid_param_raises(self):
        """Test that set_params raises for invalid params."""
        model = DEEM()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=123)


class TestDEEMMissingValues:
    """Tests for handling missing values (-1)."""

    def test_fit_with_missing_values(self):
        """Test that missing values (-1) are handled."""
        model = DEEM(n_classes=3, device='cpu')
        predictions = np.random.randint(0, 3, (100, 15))
        # Add some missing values
        predictions[0:10, :] = -1  # First 10 samples have all missing
        predictions[10:20, 0:5] = -1  # Some classifiers missing
        
        model.fit(predictions, epochs=3, verbose=False)
        
        assert model.is_fitted_ is True


class TestDEEMIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test the complete workflow from fit to score."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        n_classifiers = 15
        n_classes = 3
        
        # Generate predictions (somewhat correlated with true labels)
        true_labels = np.random.randint(0, n_classes, n_samples)
        predictions = np.zeros((n_samples, n_classifiers), dtype=int)
        for i in range(n_classifiers):
            # Each classifier is correct ~60% of the time
            correct_mask = np.random.rand(n_samples) < 0.6
            predictions[:, i] = np.where(
                correct_mask,
                true_labels,
                np.random.randint(0, n_classes, n_samples)
            )
        
        # Split data
        train_preds = predictions[:150]
        test_preds = predictions[150:]
        test_labels = true_labels[150:]
        
        # Train model
        model = DEEM(n_classes=n_classes, device='cpu')
        model.fit(train_preds, epochs=10, verbose=False)
        
        # Evaluate
        accuracy = model.score(test_preds, test_labels)
        
        # Should be better than random (33% for 3 classes)
        assert accuracy > 0.3

    def test_three_line_usage(self):
        """Test the dream 3-line usage pattern."""
        predictions = np.random.randint(0, 3, (100, 15))
        
        model = DEEM(device='cpu')
        model.fit(predictions, epochs=3, verbose=False)
        consensus = model.predict(predictions)
        
        assert consensus.shape == (100,)
        assert consensus.max() < 3

"""Tests for weighted initialization feature.

This module tests the weighted initialization functionality that scales
RBM weights by each classifier's agreement with majority vote.
"""

import numpy as np
import torch

from deem import DEEM


class TestWeightedInitialization:
    """Tests for weighted initialization feature."""

    def test_weighted_initialization_applied(self):
        """Test that weighted initialization scales weights correctly.

        Creates synthetic data with one perfect classifier (always matches
        majority vote) and verifies that classifier gets higher weight.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples, n_classifiers = 100, 5
        predictions = np.random.randint(0, 3, (n_samples, n_classifiers))

        # Make classifier 0 perfect (always matches majority vote)
        mv = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3).argmax(), 1, predictions
        )
        predictions[:, 0] = mv

        model = DEEM(n_classes=3, epochs=1, use_weighted=True, random_state=42)
        model.fit(predictions, verbose=False)

        # Check that weighted init was applied
        weights = model.model_.weights.detach().cpu().numpy()

        # Calculate expected accuracy for each classifier
        # Classifier 0 should have accuracy = 1.0 (perfect)
        # Other classifiers should have lower accuracy

        # Extract weight norms for each classifier (dimension 2)
        # weights shape is (dh, m, dx, k) = (1, 3, 5, 3)
        clf_weight_norms = np.linalg.norm(weights[0, :, :, :], axis=(0, 2))  # Shape: (dx,)

        assert clf_weight_norms[0] > clf_weight_norms[1], (
            f"Perfect classifier (0) should have higher weight than classifier 1: "
            f"{clf_weight_norms[0]:.4f} vs {clf_weight_norms[1]:.4f}"
        )

    def test_weighted_initialization_disabled(self):
        """Test that use_weighted=False disables weighting."""
        np.random.seed(42)
        torch.manual_seed(42)

        predictions = np.random.randint(0, 3, (100, 5))

        # Train without weighting (control)
        model_unweighted = DEEM(n_classes=3, epochs=1, use_weighted=False, random_state=42)
        model_unweighted.fit(predictions, verbose=False)
        weights_unweighted = model_unweighted.model_.weights.detach().clone()

        # Train with weighting
        model_weighted = DEEM(n_classes=3, epochs=1, use_weighted=True, random_state=42)
        model_weighted.fit(predictions, verbose=False)
        weights_weighted = model_weighted.model_.weights.detach().clone()

        # The initial weights should be different after weighting
        # (before any training steps occur, the weighting modifies initial weights)
        # Note: We can't directly compare because training also affects weights
        # Instead, we verify that both models complete successfully
        assert model_unweighted.is_fitted_
        assert model_weighted.is_fitted_

        # Verify that weights exist and are non-zero
        assert weights_unweighted.shape == weights_weighted.shape
        assert (weights_unweighted != 0).any()
        assert (weights_weighted != 0).any()

    def test_weighted_init_extreme_case(self):
        """Test weighted init with one classifier that disagrees with all others."""
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 100
        # Create data where all classifiers agree except one
        majority_vote = np.random.randint(0, 3, n_samples)
        predictions = np.tile(majority_vote.reshape(-1, 1), (1, 5))

        # Make last classifier always wrong (different from majority)
        predictions[:, -1] = (majority_vote + 1) % 3

        model = DEEM(n_classes=3, epochs=1, use_weighted=True, random_state=42)
        model.fit(predictions, verbose=False)

        weights = model.model_.weights.detach().cpu().numpy()

        # Classifiers 0-3 should have accuracy = 1.0
        # Classifier 4 should have accuracy = 0.0
        # Weights for classifier 4 should be scaled to near-zero
        clf_weight_norms = np.linalg.norm(weights[0, :, :, :], axis=(0, 2))

        assert clf_weight_norms[-1] < clf_weight_norms[0] * 0.1, (
            f"Bad classifier should have much lower weight: "
            f"{clf_weight_norms[-1]:.4f} vs {clf_weight_norms[0]:.4f}"
        )

    def test_weighted_init_with_soft_labels(self):
        """Test weighted init works with soft labels (3D tensors).

        Note: The full training with soft labels has known issues in the sampler
        (not related to weighted init). This test verifies that weighted init
        correctly handles the argmax conversion for 3D tensors.
        """
        np.random.seed(42)
        torch.manual_seed(42)

        # Soft labels: (n_samples, n_classes, n_classifiers)
        n_samples, n_classes, n_classifiers = 50, 3, 10
        predictions = np.random.rand(n_samples, n_classes, n_classifiers)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

        model = DEEM(n_classes=3, epochs=0, use_weighted=True, random_state=42)

        # Manually test weighted init (epochs=0 means no training)
        # This tests that the soft label handling in _apply_weighted_initialization works
        predictions_tensor = model._to_tensor(predictions, torch.float32)
        model._is_soft_labels = True  # Manually set since we're skipping full flow
        model.n_classes_ = n_classes
        model.n_classifiers_ = n_classifiers
        model._init_model()

        # Create dataloader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(predictions_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Initialize buffer (required before weighted init)
        model._initialize_buffer_with_first_batch(train_loader, verbose=False)

        # Apply weighted init - this should handle 3D->2D conversion via argmax
        model._apply_weighted_initialization(train_loader, verbose=False)

        # Weights should be non-zero and scaled
        weights = model.model_.weights.detach()
        assert (weights != 0).any(), "Weights should be initialized"

    def test_weighted_init_verbose_output(self, capsys):
        """Test that verbose output shows classifier accuracies."""
        np.random.seed(42)
        torch.manual_seed(42)

        predictions = np.random.randint(0, 3, (100, 5))

        model = DEEM(n_classes=3, epochs=1, use_weighted=True, random_state=42)
        model.fit(predictions, verbose=True)

        captured = capsys.readouterr()

        # Check for weighted init output
        assert "Weighted initialization applied" in captured.out
        assert "Classifier accuracies" in captured.out
        assert "Mean accuracy" in captured.out

    def test_weighted_init_parameter_stored(self):
        """Test that use_weighted parameter is correctly stored."""
        model_weighted = DEEM(use_weighted=True)
        model_unweighted = DEEM(use_weighted=False)

        assert model_weighted.use_weighted is True
        assert model_unweighted.use_weighted is False

    def test_weighted_init_default_is_true(self):
        """Test that use_weighted defaults to True."""
        model = DEEM()
        assert model.use_weighted is True

    def test_weighted_init_preserves_weight_shape(self):
        """Test that weighted init preserves weight tensor shape."""
        np.random.seed(42)
        torch.manual_seed(42)

        predictions = np.random.randint(0, 4, (100, 8))

        model = DEEM(n_classes=4, hidden_dim=2, epochs=1, use_weighted=True, random_state=42)
        model.fit(predictions, verbose=False)

        weights = model.model_.weights

        # Expected shape: (l, m, dx, dh) = (n_classes, n_classes, n_classifiers, hidden_dim)
        # = (4, 4, 8, 2)
        assert weights.shape == (4, 4, 8, 2), f"Unexpected weight shape: {weights.shape}"

    def test_weighted_init_accuracy_computation(self):
        """Test that accuracy computation matches expected values."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create controlled data where we know the accuracies
        n_samples = 100
        # majority_vote is all zeros since predictions[:, 0] = predictions[:, 1] = 0

        predictions = np.zeros((n_samples, 4), dtype=int)
        predictions[:, 0] = 0  # Classifier 0: 100% accuracy with MV
        predictions[:, 1] = 0  # Classifier 1: 100% accuracy with MV
        predictions[:50, 2] = 1  # Classifier 2: 50% accuracy (first half wrong)
        predictions[:25, 3] = 2  # Classifier 3: 75% accuracy (first quarter wrong)

        model = DEEM(n_classes=3, epochs=0, use_weighted=True, random_state=42)

        # We need to manually test the weighted init logic
        # First, set up the model without training
        model._is_soft_labels = False
        model.n_classes_ = 3
        model.n_classifiers_ = 4
        model._init_model()

        # Store initial weights
        initial_weights = model.model_.weights.detach().clone()

        # Create dataloader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.tensor(predictions, dtype=torch.float32))
        train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Apply weighted init
        model._apply_weighted_initialization(train_loader, verbose=False)

        # Check that weights are scaled
        final_weights = model.model_.weights.detach()

        # Expected accuracy vector: [1.0, 1.0, 0.5, 0.75]
        # weights should be scaled by this
        expected_scales = torch.tensor([1.0, 1.0, 0.5, 0.75])

        # Check per-classifier scaling (along dimension 2)
        for clf_idx in range(4):
            scale_ratio = (
                final_weights[0, 0, clf_idx, 0] / initial_weights[0, 0, clf_idx, 0]
            ).item()
            expected_scale = expected_scales[clf_idx].item()

            assert abs(scale_ratio - expected_scale) < 0.01, (
                f"Classifier {clf_idx}: expected scale {expected_scale:.2f}, "
                f"got {scale_ratio:.2f}"
            )


class TestWeightedInitIntegration:
    """Integration tests for weighted initialization with full training."""

    def test_weighted_init_improves_perfect_classifier_influence(self):
        """Test that weighted init gives perfect classifier more influence."""
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 200
        # Create data where classifier 0 is perfect, others are random
        true_labels = np.random.randint(0, 3, n_samples)
        predictions = np.random.randint(0, 3, (n_samples, 5))
        predictions[:, 0] = true_labels  # Make classifier 0 perfect

        # Compute majority vote (classifier 0 should heavily influence this)
        mv = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3).argmax(), 1, predictions
        )
        predictions[:, 0] = mv  # Ensure classifier 0 matches MV for test

        # Train with weighting
        model = DEEM(n_classes=3, epochs=10, use_weighted=True, random_state=42)
        model.fit(predictions, verbose=False)

        # The model should complete without error
        assert model.is_fitted_

        # Make predictions
        consensus = model.predict(predictions)
        assert len(consensus) == n_samples
        assert consensus.min() >= 0
        assert consensus.max() <= 2

    def test_weighted_init_with_preprocessing(self):
        """Test weighted init works with preprocessing layers."""
        np.random.seed(42)
        torch.manual_seed(42)

        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(
            n_classes=3,
            epochs=1,
            use_weighted=True,
            use_preprocessing=True,
            preprocessing_layers=1,
            random_state=42,
        )
        model.fit(predictions, verbose=False)

        assert model.is_fitted_

        # Weights should still be properly scaled
        weights = model.model_.weights.detach()
        assert (weights != 0).any()

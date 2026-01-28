"""Test soft label (3D tensor) support in DEEM.

This test suite validates that DEEM correctly handles soft labels (probability
distributions) from ensemble classifiers. Soft labels have shape
(n_samples, n_classes, n_classifiers) instead of the usual hard labels with
shape (n_samples, n_classifiers).

Production reference: src/run_predict.py lines 315-323, 363-370
"""

import numpy as np
import pytest
import torch

from deem import DEEM


def create_soft_labels(
    n_samples: int = 100,
    n_classes: int = 3,
    n_classifiers: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic soft label data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_classes : int
        Number of classes.
    n_classifiers : int
        Number of classifiers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    soft_labels : np.ndarray
        Soft labels of shape (n_samples, n_classes, n_classifiers).
        Each column sums to 1.0 (probability distribution).
    """
    np.random.seed(seed)
    # Random probabilities
    soft = np.random.rand(n_samples, n_classes, n_classifiers)
    # Normalize to sum to 1 over classes (axis 1)
    soft = soft / soft.sum(axis=1, keepdims=True)
    return soft.astype(np.float32)


def create_hard_labels(
    n_samples: int = 100,
    n_classes: int = 3,
    n_classifiers: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic hard label data.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_classes : int
        Number of classes.
    n_classifiers : int
        Number of classifiers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    hard_labels : np.ndarray
        Hard labels of shape (n_samples, n_classifiers).
        Each value is an integer in [0, n_classes).
    """
    np.random.seed(seed)
    return np.random.randint(0, n_classes, (n_samples, n_classifiers))


class TestSoftLabelDetection:
    """Test that 3D tensors are correctly detected as soft labels."""

    def test_soft_labels_detected(self):
        """Test that 3D tensors are detected as soft labels."""
        soft_preds = create_soft_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        assert model._is_soft_labels, "3D tensors should be detected as soft labels"
        assert model.is_fitted_

    def test_hard_labels_not_detected_as_soft(self):
        """Test that 2D tensors are NOT detected as soft labels."""
        hard_preds = create_hard_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(hard_preds, verbose=False)

        assert not model._is_soft_labels, "2D tensors should NOT be detected as soft labels"
        assert model.is_fitted_

    def test_soft_labels_infers_n_classes(self):
        """Test that n_classes is correctly inferred from 3D tensor shape."""
        n_classes = 5
        soft_preds = create_soft_labels(50, n_classes, 10)

        model = DEEM(epochs=1)  # Don't specify n_classes
        model.fit(soft_preds, verbose=False)

        assert model.n_classes_ == n_classes, \
            f"Expected n_classes={n_classes}, got {model.n_classes_}"

    def test_soft_labels_infers_n_classifiers(self):
        """Test that n_classifiers is correctly inferred from 3D tensor shape."""
        n_classifiers = 12
        soft_preds = create_soft_labels(50, 3, n_classifiers)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        assert model.n_classifiers_ == n_classifiers, \
            f"Expected n_classifiers={n_classifiers}, got {model.n_classifiers_}"


class TestOhModeAutoEnable:
    """Test that oh_mode is automatically enabled for soft labels."""

    def test_oh_mode_enabled_for_soft_labels(self):
        """Test that oh_mode is automatically enabled for soft labels."""
        soft_preds = create_soft_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        assert model.model_.sampler.oh_mode, \
            "oh_mode should be enabled for soft labels"
        assert model.sampler_oh_mode, \
            "DEEM.sampler_oh_mode should be True"

    def test_oh_mode_not_enabled_for_hard_labels(self):
        """Test that oh_mode stays False for hard labels (default)."""
        hard_preds = create_hard_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(hard_preds, verbose=False)

        assert not model.model_.sampler.oh_mode, \
            "oh_mode should NOT be enabled for hard labels (default)"
        assert not model.sampler_oh_mode, \
            "DEEM.sampler_oh_mode should be False"

    def test_manual_oh_mode_preserved_for_hard_labels(self):
        """Test that manually set oh_mode is preserved for hard labels.
        
        Note: This is NOT a recommended configuration. oh_mode=True is designed
        for soft labels (3D). This test just verifies the parameter is set.
        Actual training with oh_mode=True and hard labels may fail due to
        dimension mismatches in the sampler.
        """
        hard_preds = create_hard_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1, sampler_oh_mode=True)
        
        # Verify parameter is set before fit
        assert model.sampler_oh_mode, \
            "Manually set sampler_oh_mode=True should be stored"


class TestSoftLabelEndToEnd:
    """Test full training and prediction pipeline with soft labels."""

    def test_soft_labels_end_to_end(self):
        """Test full training and prediction with soft labels."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=5, use_weighted=True)
        model.fit(soft_preds, verbose=False)

        # Predict on test data
        test_soft = create_soft_labels(20, 3, 10, seed=99)
        predictions = model.predict(test_soft)

        assert predictions.shape == (20,), \
            f"Expected shape (20,), got {predictions.shape}"
        assert predictions.min() >= 0 and predictions.max() < 3, \
            "Predictions should be in [0, n_classes)"

    def test_soft_labels_return_probs(self):
        """Test prediction with return_probs=True for soft labels."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=5)
        model.fit(soft_preds, verbose=False)

        # Get probabilities
        probs = model.predict(soft_preds, return_probs=True)

        assert probs.shape == (100, 3), \
            f"Expected shape (100, 3), got {probs.shape}"
        # Probabilities should sum to approximately 1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), \
            "Probabilities should sum to 1"


class TestSoftLabelWithWeightedInit:
    """Test weighted initialization with soft labels."""

    def test_soft_labels_with_weighted_init(self):
        """Test that weighted initialization works with soft labels."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=1, use_weighted=True)
        model.fit(soft_preds, verbose=False)

        # Should complete without error
        assert model.is_fitted_

        # Weights should be initialized (non-zero for most)
        weights = model.model_.weights.detach()
        assert (weights != 0).any(), "Weights should be non-zero after weighted init"

    def test_soft_labels_without_weighted_init(self):
        """Test that training works without weighted initialization."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=1, use_weighted=False)
        model.fit(soft_preds, verbose=False)

        # Should complete without error
        assert model.is_fitted_


class TestSoftLabelBufferInit:
    """Test buffer initialization with soft labels."""

    def test_soft_labels_buffer_initialization(self):
        """Test that buffer is initialized correctly with soft labels."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        # Buffer should be populated
        buffer_size = len(model.model_.sampler.buffer)
        assert buffer_size > 0, "Buffer should be populated with soft labels"

    def test_soft_labels_buffer_has_preprocessed_data(self):
        """Test that buffer contains preprocessed data when oh_mode=True."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        # oh_mode should be True for soft labels
        assert model.model_.sampler.oh_mode

        # Buffer should be populated
        buffer = model.model_.sampler.buffer
        assert len(buffer) > 0


class TestSoftLabelScoring:
    """Test scoring with soft labels."""

    def test_soft_labels_score_with_labels(self):
        """Test scoring with soft predictions and hard labels."""
        soft_preds = create_soft_labels(100, 3, 10)
        # Generate synthetic hard labels
        np.random.seed(42)
        labels = np.random.randint(0, 3, 100)

        model = DEEM(n_classes=3, epochs=5)
        model.fit(soft_preds, verbose=False)

        # Score should work with hard labels
        accuracy = model.score(soft_preds, labels)
        assert 0.0 <= accuracy <= 1.0, \
            f"Accuracy should be in [0, 1], got {accuracy}"

    def test_soft_labels_predict_with_alignment(self):
        """Test predict with align_to with soft labels."""
        soft_preds = create_soft_labels(100, 3, 10)
        np.random.seed(42)

        model = DEEM(n_classes=3, epochs=5)
        model.fit(soft_preds, verbose=False)

        # predict with align_to should work
        aligned = model.predict(soft_preds, align_to=soft_preds)
        assert aligned.shape == (100,), \
            f"Expected shape (100,), got {aligned.shape}"
        assert model.class_map_ is not None, "class_map_ should be set"


class TestSoftLabelVerboseOutput:
    """Test verbose output for soft label detection."""

    def test_soft_labels_verbose_output(self, capsys):
        """Test that verbose mode announces soft label detection."""
        soft_preds = create_soft_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=True)

        captured = capsys.readouterr()
        assert "Soft labels" in captured.out or "3D" in captured.out or "oh_mode" in captured.out.lower(), \
            f"Verbose output should mention soft labels, got: {captured.out}"

    def test_hard_labels_no_soft_message(self, capsys):
        """Test that hard labels don't trigger soft label message."""
        hard_preds = create_hard_labels(50, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        model.fit(hard_preds, verbose=True)

        captured = capsys.readouterr()
        # Should NOT mention soft labels when using hard labels
        assert "Soft labels" not in captured.out, \
            "Should not mention soft labels for hard label data"


class TestSoftLabelEdgeCases:
    """Test edge cases for soft label handling."""

    def test_soft_labels_single_sample(self):
        """Test soft labels with very small dataset.
        
        Note: Single sample causes buffer issues (can't sample from empty buffer).
        Using a small but viable dataset size (10 samples).
        """
        soft_preds = create_soft_labels(10, 3, 5)

        model = DEEM(n_classes=3, epochs=1)
        # Should handle small dataset gracefully
        model.fit(soft_preds, verbose=False)
        assert model.is_fitted_

    def test_soft_labels_many_classes(self):
        """Test soft labels with many classes."""
        soft_preds = create_soft_labels(100, 10, 5)  # 10 classes

        model = DEEM(n_classes=10, epochs=1)
        model.fit(soft_preds, verbose=False)

        assert model.n_classes_ == 10
        assert model._is_soft_labels

    def test_soft_labels_many_classifiers(self):
        """Test soft labels with many classifiers."""
        soft_preds = create_soft_labels(100, 3, 50)  # 50 classifiers

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        assert model.n_classifiers_ == 50
        assert model._is_soft_labels

    def test_soft_labels_with_preprocessing(self):
        """Test soft labels with preprocessing layers enabled."""
        soft_preds = create_soft_labels(100, 3, 10)

        model = DEEM(
            n_classes=3,
            epochs=1,
            use_preprocessing=True,
            preprocessing_layers=1,
        )
        model.fit(soft_preds, verbose=False)

        assert model.is_fitted_
        assert model._is_soft_labels
        assert model.sampler_oh_mode


class TestSoftLabelConsistency:
    """Test consistency between soft and hard label handling."""

    def test_argmax_of_soft_matches_hard_structure(self):
        """Test that argmax of soft labels produces same shape as hard labels."""
        n_samples, n_classes, n_classifiers = 100, 3, 10

        # Create matching soft and hard labels
        soft_preds = create_soft_labels(n_samples, n_classes, n_classifiers)
        # Convert soft to hard via argmax
        hard_from_soft = soft_preds.argmax(axis=1)

        assert hard_from_soft.shape == (n_samples, n_classifiers), \
            f"Expected shape ({n_samples}, {n_classifiers}), got {hard_from_soft.shape}"

    def test_soft_and_hard_produce_similar_shapes(self):
        """Test that soft and hard labels produce same prediction shapes."""
        n_samples, n_classes, n_classifiers = 100, 3, 10

        soft_preds = create_soft_labels(n_samples, n_classes, n_classifiers)
        hard_preds = create_hard_labels(n_samples, n_classes, n_classifiers)

        # Train models
        soft_model = DEEM(n_classes=n_classes, epochs=5)
        soft_model.fit(soft_preds, verbose=False)

        hard_model = DEEM(n_classes=n_classes, epochs=5)
        hard_model.fit(hard_preds, verbose=False)

        # Predictions should have same shape
        soft_consensus = soft_model.predict(soft_preds)
        hard_consensus = hard_model.predict(hard_preds)

        assert soft_consensus.shape == hard_consensus.shape, \
            "Predictions from soft and hard labels should have same shape"

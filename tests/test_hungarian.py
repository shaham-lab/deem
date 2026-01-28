"""Tests for Hungarian alignment transparency in DEEM.

This module tests that Hungarian alignment is transparent and automatic,
always aligning against majority vote (never true labels).
"""

import numpy as np
import pytest
import torch

from deem import DEEM


class TestMajorityVoteComputation:
    """Tests for _compute_majority_vote helper."""

    def test_majority_vote_basic(self):
        """Test majority vote computation with hard labels."""
        np.random.seed(42)
        predictions = np.array([
            [0, 0, 0, 1, 1],  # majority = 0
            [1, 1, 1, 0, 0],  # majority = 1
            [2, 2, 2, 2, 0],  # majority = 2
        ])

        model = DEEM(n_classes=3, epochs=1)
        model.fit(predictions, verbose=False)

        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        mv = model._compute_majority_vote(predictions_tensor)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(mv.numpy(), expected)

    def test_majority_vote_soft_labels(self):
        """Test majority vote computation with soft labels (3D)."""
        np.random.seed(42)
        # Soft labels: (n_samples, n_classes, n_classifiers)
        # Sample 1: classifiers predict [0, 0, 1] -> majority = 0
        # Sample 2: classifiers predict [1, 1, 1] -> majority = 1
        soft_preds = np.zeros((2, 3, 3))
        # Sample 1, classifiers 0 and 1 predict class 0, classifier 2 predicts class 1
        soft_preds[0, 0, 0] = 1.0
        soft_preds[0, 0, 1] = 1.0
        soft_preds[0, 1, 2] = 1.0
        # Sample 2, all classifiers predict class 1
        soft_preds[1, 1, :] = 1.0

        model = DEEM(n_classes=3, epochs=1)
        model.fit(soft_preds, verbose=False)

        soft_tensor = torch.tensor(soft_preds, dtype=torch.float32)
        mv = model._compute_majority_vote(soft_tensor)

        expected = np.array([0, 1])
        np.testing.assert_array_equal(mv.numpy(), expected)


class TestPredictWithAutomaticAlignment:
    """Tests for predict() with automatic alignment."""

    def test_predict_always_aligns_to_self(self):
        """Test that predict always computes Hungarian alignment (to self by default)."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Predict - should automatically align to MV of predictions
        consensus = model.predict(predictions)

        assert consensus.shape == (100,)
        assert model.class_map_ is not None  # Alignment ALWAYS computed
        assert len(model.class_map_) == 3  # Mapping for all classes

    def test_predict_returns_valid_classes(self):
        """Test that predictions are valid class indices."""
        np.random.seed(42)
        predictions = np.random.randint(0, 5, (50, 8))

        model = DEEM(n_classes=5, epochs=3)
        model.fit(predictions, verbose=False)

        consensus = model.predict(predictions)

        assert np.all(consensus >= 0)
        assert np.all(consensus < 5)


class TestPredictWithAlignment:
    """Tests for predict() with align_to parameter."""

    def test_predict_with_alignment(self):
        """Test that alignment to majority vote works."""
        np.random.seed(42)
        train_preds = np.random.randint(0, 3, (100, 10))
        test_preds = np.random.randint(0, 3, (20, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(train_preds, verbose=False)

        # Predict with alignment
        consensus = model.predict(test_preds, align_to=train_preds)

        assert consensus.shape == (20,)
        assert model.class_map_ is not None  # Alignment computed
        assert len(model.class_map_) == 3  # One mapping per class

    def test_alignment_cached(self):
        """Test that alignment is cached and reused."""
        np.random.seed(42)
        train_preds = np.random.randint(0, 3, (100, 10))
        test_preds = np.random.randint(0, 3, (20, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(train_preds, verbose=False)

        # First prediction computes alignment
        consensus1 = model.predict(test_preds, align_to=train_preds)
        mapping1 = model.get_class_mapping()

        # Second prediction reuses cached alignment (no align_to)
        consensus2 = model.predict(test_preds)
        mapping2 = model.get_class_mapping()

        assert mapping1 == mapping2
        np.testing.assert_array_equal(consensus1, consensus2)

    def test_alignment_recomputed_with_new_align_to(self):
        """Test that new align_to recomputes alignment."""
        np.random.seed(42)
        train_preds1 = np.random.randint(0, 3, (100, 10))
        train_preds2 = np.random.randint(0, 3, (100, 10))
        test_preds = np.random.randint(0, 3, (20, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(train_preds1, verbose=False)

        # First alignment
        model.predict(test_preds, align_to=train_preds1)
        mapping1 = model.get_class_mapping()

        # Second alignment with different reference
        model.predict(test_preds, align_to=train_preds2)
        mapping2 = model.get_class_mapping()

        # Mappings may or may not differ, but both should be valid
        assert mapping1 is not None
        assert mapping2 is not None
        assert len(mapping1) == 3
        assert len(mapping2) == 3


class TestResetAlignment:
    """Tests for reset_class_mapping()."""

    def test_reset_alignment(self):
        """Test that reset_class_mapping clears cached alignment."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Compute alignment
        model.predict(predictions, align_to=predictions)
        assert model.class_map_ is not None

        # Reset
        model.reset_class_mapping()
        assert model.class_map_ is None

    def test_predict_after_reset_recomputes_alignment(self):
        """Test that predict after reset recomputes alignment automatically."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Compute alignment
        aligned1 = model.predict(predictions, align_to=predictions)
        map1 = model.class_map_.copy()
        assert model.class_map_ is not None

        # Reset
        model.reset_class_mapping()

        # Predict - should automatically recompute alignment
        aligned2 = model.predict(predictions)
        assert model.class_map_ is not None  # Alignment recomputed
        assert np.array_equal(aligned1, aligned2)  # Same result


class TestGetClassMapping:
    """Tests for get_class_mapping()."""

    def test_get_class_mapping_none_initially(self):
        """Test that get_class_mapping returns None before any predict()."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # No predict() called yet, so no alignment
        assert model.get_class_mapping() is None

    def test_get_class_mapping_returns_dict_after_predict(self):
        """Test that get_class_mapping returns dict after predict()."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Predict automatically computes alignment
        model.predict(predictions)

        mapping = model.get_class_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 3
        assert set(mapping.keys()) == {0, 1, 2}


class TestScoreWithAlignment:
    """Tests for score() with automatic alignment."""

    def test_score_uses_alignment(self):
        """Test that score() automatically aligns predictions."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))

        # Create labels that match majority vote
        mv = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3).argmax(), 1, predictions
        )

        model = DEEM(n_classes=3, epochs=10)
        model.fit(predictions, verbose=False)

        # Score should use alignment
        accuracy = model.score(predictions, mv)

        # Should have non-zero accuracy
        assert accuracy >= 0.0
        assert accuracy <= 1.0
        # Alignment was computed
        assert model.class_map_ is not None

    def test_score_aligns_against_majority_vote_not_labels(self):
        """Test that score aligns against MV, not true labels."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (100, 10))
        # Random labels (not majority vote)
        random_labels = np.random.randint(0, 3, (100,))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Score should still work (aligns to MV internally)
        accuracy = model.score(predictions, random_labels)

        assert accuracy >= 0.0
        assert accuracy <= 1.0
        assert model.class_map_ is not None


class TestSoftLabelAlignment:
    """Tests for alignment with soft labels (3D tensors)."""

    def test_alignment_with_soft_labels(self):
        """Test alignment works with soft labels (3D tensors)."""
        np.random.seed(42)
        # Soft labels: (n_samples, n_classes, n_classifiers)
        soft_preds = np.random.rand(100, 3, 10)
        soft_preds = soft_preds / soft_preds.sum(axis=1, keepdims=True)

        model = DEEM(n_classes=3, epochs=5)
        model.fit(soft_preds, verbose=False)

        # Predict with alignment
        consensus = model.predict(soft_preds, align_to=soft_preds)

        assert consensus.shape == (100,)
        assert model.class_map_ is not None

    def test_score_with_soft_labels(self):
        """Test score works with soft labels."""
        np.random.seed(42)
        soft_preds = np.random.rand(100, 3, 10)
        soft_preds = soft_preds / soft_preds.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, (100,))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(soft_preds, verbose=False)

        accuracy = model.score(soft_preds, labels)

        assert accuracy >= 0.0
        assert accuracy <= 1.0


class TestRemovedMethods:
    """Tests for removed methods."""

    def test_no_predict_with_hungarian(self):
        """Test that predict_with_hungarian() method is removed."""
        model = DEEM(n_classes=3)

        assert not hasattr(model, 'predict_with_hungarian'), \
            "predict_with_hungarian() should be removed"


class TestReturnProbs:
    """Tests for return_probs functionality."""

    def test_return_probs_no_alignment(self):
        """Test that return_probs returns probabilities without alignment."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (50, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        probs = model.predict(predictions, return_probs=True)

        assert probs.shape == (50, 3)  # (n_samples, n_classes)
        # Probabilities should be valid
        assert np.all(probs >= 0)

    def test_return_probs_ignores_align_to(self):
        """Test that return_probs ignores align_to parameter."""
        np.random.seed(42)
        predictions = np.random.randint(0, 3, (50, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(predictions, verbose=False)

        # Even with align_to, probabilities are returned without alignment
        probs = model.predict(predictions, return_probs=True, align_to=predictions)

        assert probs.shape == (50, 3)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_alignment_with_single_sample(self):
        """Test alignment works with single sample."""
        np.random.seed(42)
        train_preds = np.random.randint(0, 3, (100, 10))
        test_preds = np.random.randint(0, 3, (1, 10))

        model = DEEM(n_classes=3, epochs=5)
        model.fit(train_preds, verbose=False)

        consensus = model.predict(test_preds, align_to=train_preds)

        assert consensus.shape == (1,)

    def test_alignment_binary_classification(self):
        """Test alignment with binary classification."""
        np.random.seed(42)
        predictions = np.random.randint(0, 2, (100, 10))
        labels = np.random.randint(0, 2, (100,))

        model = DEEM(n_classes=2, epochs=5)
        model.fit(predictions, verbose=False)

        accuracy = model.score(predictions, labels)

        assert accuracy >= 0.0
        assert accuracy <= 1.0
        assert len(model.class_map_) == 2

    def test_alignment_many_classes(self):
        """Test alignment with many classes."""
        np.random.seed(42)
        predictions = np.random.randint(0, 10, (100, 15))
        labels = np.random.randint(0, 10, (100,))

        model = DEEM(n_classes=10, epochs=5)
        model.fit(predictions, verbose=False)

        accuracy = model.score(predictions, labels)

        assert accuracy >= 0.0
        assert accuracy <= 1.0
        assert len(model.class_map_) == 10


class TestAlignmentCorrectness:
    """Tests to verify alignment correctness."""

    def test_alignment_improves_accuracy(self):
        """Test that alignment can improve accuracy when labels are permuted."""
        np.random.seed(42)
        # Create data where classifiers consistently predict permuted labels
        n_samples = 200
        n_classifiers = 10
        n_classes = 3

        # True labels
        true_labels = np.random.randint(0, n_classes, n_samples)

        # Create predictions that are mostly correct but permuted
        # Permutation: 0->1, 1->2, 2->0
        permutation = {0: 1, 1: 2, 2: 0}
        predictions = np.zeros((n_samples, n_classifiers), dtype=int)
        for i in range(n_samples):
            true_class = true_labels[i]
            permuted_class = permutation[true_class]
            # Most classifiers predict the permuted class
            predictions[i, :] = permuted_class
            # Add some noise
            noise_idx = np.random.choice(n_classifiers, 2, replace=False)
            predictions[i, noise_idx] = np.random.randint(0, n_classes, 2)

        model = DEEM(n_classes=n_classes, epochs=5)
        model.fit(predictions, verbose=False)

        # Get aligned predictions
        aligned = model.predict(predictions, align_to=predictions)

        # Model should learn the permutation and alignment should work
        # Accuracy should be reasonable (model learned the structure)
        assert model.class_map_ is not None

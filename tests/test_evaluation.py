"""Tests for evaluation module: Hungarian algorithm and metrics."""

import numpy as np
import pytest
import torch

from deem.core.evaluation import (
    get_hungarian_solution,
    vectorize_predictions,
    build_cost_matrix,
    accuracy_with_hungarian,
    compute_mean_std,
    extract_dead_classes_indices,
)


class TestHungarianAlgorithm:
    """Tests for Hungarian algorithm implementation."""

    def test_perfect_alignment(self):
        """Test when predictions already match true labels."""
        preds = torch.tensor([0, 0, 1, 1, 2, 2])
        true = torch.tensor([0, 0, 1, 1, 2, 2])

        class_map = get_hungarian_solution(preds, true, k=3)
        aligned = vectorize_predictions(preds, class_map)

        # Should map to itself
        assert class_map == {0: 0, 1: 1, 2: 2}
        assert np.array_equal(aligned, true.numpy())

    def test_permuted_labels(self):
        """Test with completely permuted labels."""
        preds = torch.tensor([2, 2, 1, 1, 0, 0])  # Permuted
        true = torch.tensor([0, 0, 1, 1, 2, 2])   # True labels

        class_map = get_hungarian_solution(preds, true, k=3)
        aligned = vectorize_predictions(preds, class_map)

        # Should recover correct labels
        assert np.array_equal(aligned, true.numpy())

    def test_partial_alignment(self):
        """Test with partial permutation."""
        # Classes 0 and 2 are swapped
        preds = torch.tensor([2, 2, 1, 1, 0, 0])
        true = torch.tensor([0, 0, 1, 1, 2, 2])

        class_map = get_hungarian_solution(preds, true, k=3)
        aligned = vectorize_predictions(preds, class_map)

        # Should get 100% accuracy after alignment
        accuracy = np.mean(aligned == true.numpy())
        assert accuracy == 1.0

    def test_numpy_input(self):
        """Test that numpy arrays work as input."""
        preds = np.array([1, 1, 0, 0])
        true = np.array([0, 0, 1, 1])

        class_map = get_hungarian_solution(preds, true, k=2)
        aligned = vectorize_predictions(preds, class_map)

        assert np.array_equal(aligned, true)

    def test_with_noise(self):
        """Test with noisy predictions (not perfectly separable)."""
        # Some misclassifications
        preds = torch.tensor([2, 2, 2, 1, 1, 0, 0, 0])
        true = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])

        class_map = get_hungarian_solution(preds, true, k=3)
        aligned = vectorize_predictions(preds, class_map)

        # Should still do reasonably well
        accuracy = np.mean(aligned == true.numpy())
        assert accuracy >= 0.5  # Better than random

    def test_squeezed_dimensions(self):
        """Test with extra dimensions that need squeezing."""
        preds = torch.tensor([[0], [0], [1], [1]])
        true = torch.tensor([[1], [1], [0], [0]])

        class_map = get_hungarian_solution(preds, true, k=2)
        aligned = vectorize_predictions(preds, class_map)

        # Should handle extra dimension
        assert len(aligned) == 4


class TestBuildCostMatrix:
    """Tests for cost matrix construction."""

    def test_cost_matrix_shape(self):
        """Test cost matrix has correct shape."""
        preds = torch.tensor([0, 0, 1, 1, 2, 2])
        true = torch.tensor([0, 0, 1, 1, 2, 2])

        cost = build_cost_matrix(preds, true, k=3)

        assert cost.shape == (3, 3)

    def test_cost_matrix_diagonal(self):
        """Test diagonal is zero when predictions match."""
        preds = torch.tensor([0, 0, 1, 1, 2, 2])
        true = torch.tensor([0, 0, 1, 1, 2, 2])

        cost = build_cost_matrix(preds, true, k=3)

        # Diagonal should be 0 (no disagreement when mapped to same class)
        assert cost[0, 0] == 0
        assert cost[1, 1] == 0
        assert cost[2, 2] == 0


class TestAccuracyWithHungarian:
    """Tests for accuracy_with_hungarian metric."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy after Hungarian alignment."""
        preds = np.array([2, 2, 1, 1, 0, 0])
        true = np.array([0, 0, 1, 1, 2, 2])

        accuracy = accuracy_with_hungarian(preds, true, k=3)

        assert accuracy == 1.0

    def test_return_mapping(self):
        """Test that return_mapping option works."""
        preds = np.array([1, 1, 0, 0])
        true = np.array([0, 0, 1, 1])

        accuracy, class_map = accuracy_with_hungarian(
            preds, true, k=2, return_mapping=True
        )

        assert accuracy == 1.0
        assert isinstance(class_map, dict)
        assert len(class_map) == 2

    def test_partial_accuracy(self):
        """Test with imperfect predictions."""
        # 2 wrong out of 6
        preds = np.array([0, 0, 1, 0, 2, 2])
        true = np.array([0, 0, 1, 1, 2, 2])

        accuracy = accuracy_with_hungarian(preds, true, k=3)

        assert 0.6 < accuracy < 1.0

    def test_torch_input(self):
        """Test with torch tensors."""
        preds = torch.tensor([0, 0, 1, 1])
        true = torch.tensor([1, 1, 0, 0])

        accuracy = accuracy_with_hungarian(preds, true, k=2)

        assert accuracy == 1.0


class TestComputeMeanStd:
    """Tests for compute_mean_std function."""

    def test_basic_computation(self):
        """Test basic mean and std computation."""
        accuracies = [0.8, 0.9, 0.85, 0.87, 0.88]

        mean, std = compute_mean_std(accuracies)

        assert 0.85 < mean < 0.90
        assert 0.0 < std < 0.1

    def test_identical_values(self):
        """Test with all identical values."""
        accuracies = [0.9, 0.9, 0.9, 0.9]

        mean, std = compute_mean_std(accuracies)

        assert mean == 0.9
        assert std == 0.0

    def test_verbose_output(self, capsys):
        """Test verbose output is printed."""
        accuracies = [0.8, 0.9]

        compute_mean_std(accuracies, verbose=True)

        captured = capsys.readouterr()
        assert "Mean accuracy" in captured.out


class TestExtractDeadClassesIndices:
    """Tests for dead class detection."""

    def test_no_dead_classes(self):
        """Test when all classes are well-represented."""
        conf = np.array([
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100]
        ])
        class_map = {0: 0, 1: 1, 2: 2}

        dead = extract_dead_classes_indices(conf, class_map)

        assert len(dead) == 0

    def test_one_dead_class(self):
        """Test when one class is dead."""
        conf = np.array([
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 2]  # Class 2 has very few predictions
        ])
        class_map = {0: 0, 1: 1, 2: 2}

        dead = extract_dead_classes_indices(conf, class_map)

        assert 2 in dead

    def test_custom_threshold(self):
        """Test with custom dead threshold."""
        conf = np.array([
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 50]
        ])
        class_map = {0: 0, 1: 1, 2: 2}

        # With high threshold, class 2 should be considered dead
        dead = extract_dead_classes_indices(conf, class_map, dead_threshold=60)

        assert 2 in dead

        # With low threshold, no class should be dead
        dead_low = extract_dead_classes_indices(conf, class_map, dead_threshold=10)

        assert len(dead_low) == 0


class TestIntegration:
    """Integration tests for the full evaluation pipeline."""

    def test_full_pipeline(self):
        """Test complete evaluation workflow."""
        # Simulate RBM predictions (permuted)
        n_samples = 100
        n_classes = 3

        # True labels: 33 samples per class
        true_labels = np.repeat([0, 1, 2], [33, 33, 34])

        # Predictions: class 0 -> 2, class 1 -> 0, class 2 -> 1
        preds = np.where(true_labels == 0, 2,
                        np.where(true_labels == 1, 0, 1))

        # Add some noise (5% error)
        noise_mask = np.random.RandomState(42).rand(n_samples) < 0.05
        preds[noise_mask] = np.random.RandomState(42).randint(0, 3, size=noise_mask.sum())

        # Get Hungarian solution
        class_map = get_hungarian_solution(
            torch.from_numpy(preds),
            torch.from_numpy(true_labels),
            k=n_classes
        )

        # Apply mapping
        aligned = vectorize_predictions(torch.from_numpy(preds), class_map)

        # Compute accuracy
        accuracy = np.mean(aligned == true_labels)

        # Should have high accuracy after alignment
        assert accuracy >= 0.90

    def test_multiple_runs_statistics(self):
        """Test computing statistics over multiple runs."""
        # Simulate 5 runs with different accuracies
        accuracies = [0.85, 0.87, 0.86, 0.88, 0.84]

        mean, std = compute_mean_std(accuracies)

        # Check reasonable values
        assert 0.84 < mean < 0.88
        assert std < 0.05

    def test_gpu_to_cpu_handling(self):
        """Test that GPU tensors are handled correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        preds = torch.tensor([1, 1, 0, 0]).cuda()
        true = torch.tensor([0, 0, 1, 1]).cuda()

        # Should automatically move to CPU
        class_map = get_hungarian_solution(preds, true, k=2)
        aligned = vectorize_predictions(preds, class_map)

        assert isinstance(aligned, np.ndarray)
        assert np.array_equal(aligned, true.cpu().numpy())

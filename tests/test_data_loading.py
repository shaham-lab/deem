"""Tests for deem.data module."""

import pytest
import torch
import numpy as np
import tempfile
import scipy.io
from pathlib import Path

from deem.data import (
    load_dataset,
    load_from_mat,
    load_from_mat_soft,
    TensorDatasetWithShape,
    convert_soft_to_hard,
    filter_missing_samples
)


class TestTensorDatasetWithShape:
    """Tests for TensorDatasetWithShape class."""
    
    def test_hard_labels_shape(self):
        """Test TensorDatasetWithShape with hard labels"""
        data = torch.randint(0, 3, (100, 15))
        labels = torch.randint(0, 3, (100,))
        dataset = TensorDatasetWithShape(data, labels)
        
        assert dataset.shape == (100, 15)
        assert len(dataset) == 100

    def test_soft_labels_shape(self):
        """Test TensorDatasetWithShape with soft labels"""
        data = torch.rand(100, 3, 15)
        labels = torch.randint(0, 3, (100,))
        dataset = TensorDatasetWithShape(data, labels)
        
        assert dataset.shape == (100, 3, 15)
        assert len(dataset) == 100

    def test_repr(self):
        """Test string representation"""
        data = torch.rand(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDatasetWithShape(data, labels)
        
        repr_str = repr(dataset)
        assert "TensorDatasetWithShape" in repr_str
        assert "50" in repr_str


class TestConvertSoftToHard:
    """Tests for convert_soft_to_hard function."""
    
    def test_basic_conversion(self):
        """Test soft to hard label conversion"""
        # Create soft label dataset: (N, K, D) = (50, 3, 10)
        soft_data = torch.rand(50, 3, 10)
        labels = torch.randint(0, 3, (50,))
        dataset = TensorDatasetWithShape(soft_data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # Convert
        hard_loader = convert_soft_to_hard(loader)
        
        # Check shape: should now be (batch_size, D)
        batch, _ = next(iter(hard_loader))
        assert batch.shape == (10, 10)

    def test_argmax_correctness(self):
        """Test that argmax is computed correctly"""
        # Create predictable soft labels
        soft_data = torch.zeros(10, 3, 5)
        soft_data[:, 2, :] = 1.0  # Class 2 has highest prob
        labels = torch.zeros(10).long()
        
        dataset = TensorDatasetWithShape(soft_data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        hard_loader = convert_soft_to_hard(loader)
        batch, _ = next(iter(hard_loader))
        
        # All predictions should be class 2
        assert (batch == 2).all()

    def test_invalid_input_raises(self):
        """Test that 2D input raises ValueError"""
        data = torch.rand(50, 10)  # 2D, not 3D
        labels = torch.randint(0, 3, (50,))
        dataset = TensorDatasetWithShape(data, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        with pytest.raises(ValueError, match="Expected 3D input"):
            convert_soft_to_hard(loader)


class TestFilterMissingSamples:
    """Tests for filter_missing_samples function."""
    
    def test_filter_all_missing_2d(self):
        """Test filtering samples with all missing values (2D)"""
        data = np.array([
            [0, 1, 2],   # Valid
            [-1, -1, -1],  # All missing
            [1, 0, 1],   # Valid
            [-1, -1, -1],  # All missing
        ])
        labels = np.array([0, 1, 2, 0])
        
        filtered_data, filtered_labels = filter_missing_samples(data, labels)
        
        assert len(filtered_data) == 2
        assert len(filtered_labels) == 2
        np.testing.assert_array_equal(filtered_labels, [0, 2])

    def test_filter_partial_missing(self):
        """Test that partial missing values are kept"""
        data = np.array([
            [-1, 1, 2],   # Partial missing - keep
            [-1, -1, -1],  # All missing - remove
            [1, -1, 1],   # Partial missing - keep
        ])
        labels = np.array([0, 1, 2])
        
        filtered_data, filtered_labels = filter_missing_samples(data, labels)
        
        assert len(filtered_data) == 2
        np.testing.assert_array_equal(filtered_labels, [0, 2])

    def test_filter_3d_data(self):
        """Test filtering 3D soft labels"""
        data = np.zeros((4, 3, 5))
        data[0] = np.random.rand(3, 5)  # Valid
        data[1] = -1  # All missing
        data[2] = np.random.rand(3, 5)  # Valid
        data[3] = -1  # All missing
        labels = np.array([0, 1, 2, 0])
        
        filtered_data, filtered_labels = filter_missing_samples(data, labels)
        
        assert len(filtered_data) == 2

    def test_tensor_input_output(self):
        """Test that tensor input returns tensor output"""
        data = torch.tensor([
            [0, 1, 2],
            [-1, -1, -1],
            [1, 0, 1],
        ])
        labels = torch.tensor([0, 1, 2])
        
        filtered_data, filtered_labels = filter_missing_samples(data, labels)
        
        assert isinstance(filtered_data, torch.Tensor)
        assert isinstance(filtered_labels, torch.Tensor)
        assert len(filtered_data) == 2


class TestLoadFromMat:
    """Tests for load_from_mat function."""
    
    @pytest.fixture
    def hard_mat_file(self, tmp_path):
        """Create a temporary .mat file with hard labels"""
        mat_file = tmp_path / "test_hard.mat"
        data = {
            'f': np.random.randint(0, 3, (100, 15)),
            'y': np.random.randint(0, 3, (100, 1)),
            'k': 3
        }
        scipy.io.savemat(mat_file, data)
        return str(mat_file)
    
    def test_load_basic(self, hard_mat_file):
        """Test basic loading of hard labels"""
        train, val, test, meta = load_from_mat(hard_mat_file)
        
        assert meta['format'] == 'hard_labels'
        assert meta['n_classifiers'] == 15
        assert meta['n_classes'] == 3
        assert len(meta['shape']) == 2

    def test_loader_shapes(self, hard_mat_file):
        """Test that loaders return correct shapes"""
        train, val, test, meta = load_from_mat(hard_mat_file, batch_size=16)
        
        batch_x, batch_y = next(iter(train))
        assert batch_x.shape[1] == 15  # n_classifiers
        assert len(batch_y.shape) == 1

    def test_combined_dataset(self, hard_mat_file):
        """Test combined_dataset mode"""
        train, val, test, meta = load_from_mat(
            hard_mat_file, 
            combined_dataset=True
        )
        
        # All loaders should have same underlying data
        train_size = sum(len(b[0]) for b in train)
        test_size = sum(len(b[0]) for b in test)
        assert train_size == test_size


class TestLoadFromMatSoft:
    """Tests for load_from_mat_soft function."""
    
    @pytest.fixture
    def soft_mat_file(self, tmp_path):
        """Create a temporary .mat file with soft labels"""
        mat_file = tmp_path / "test_soft.mat"
        data = {
            'f': np.random.rand(100, 3, 15),  # (N, K, D)
            'y': np.random.randint(0, 3, (100, 1)),
            'k': 3
        }
        scipy.io.savemat(mat_file, data)
        return str(mat_file)
    
    def test_load_basic(self, soft_mat_file):
        """Test basic loading of soft labels"""
        train, val, test, meta = load_from_mat_soft(soft_mat_file)
        
        assert meta['format'] == 'soft_labels'
        assert meta['n_classifiers'] == 15
        assert meta['n_classes'] == 3
        assert len(meta['shape']) == 3

    def test_loader_shapes(self, soft_mat_file):
        """Test that loaders return 3D data"""
        train, val, test, meta = load_from_mat_soft(soft_mat_file, batch_size=16)
        
        batch_x, batch_y = next(iter(train))
        assert len(batch_x.shape) == 3  # (batch, K, D)
        assert batch_x.shape[1] == 3    # K classes
        assert batch_x.shape[2] == 15   # D classifiers


class TestLoadDataset:
    """Tests for unified load_dataset function."""
    
    def test_auto_detect_hard_tensor(self):
        """Test automatic detection of hard labels from tensor"""
        data = torch.randint(0, 3, (100, 15))
        labels = torch.randint(0, 3, (100,))
        
        train, val, test, meta = load_dataset(data, labels)
        
        assert meta['format'] == 'hard_labels'
        assert meta['shape'] == (100, 15)

    def test_auto_detect_soft_tensor(self):
        """Test automatic detection of soft labels from tensor"""
        data = torch.rand(100, 3, 15)
        labels = torch.randint(0, 3, (100,))
        
        train, val, test, meta = load_dataset(data, labels)
        
        assert meta['format'] == 'soft_labels'
        assert meta['shape'] == (100, 3, 15)

    def test_numpy_array_input(self):
        """Test loading from numpy arrays"""
        data = np.random.randint(0, 3, (50, 10))
        labels = np.random.randint(0, 3, (50,))
        
        train, val, test, meta = load_dataset(data, labels)
        
        assert meta['format'] == 'hard_labels'

    def test_mat_file_auto_detect(self, tmp_path):
        """Test auto-detection from .mat file"""
        # Hard labels file
        hard_file = tmp_path / "hard.mat"
        scipy.io.savemat(hard_file, {
            'f': np.random.randint(0, 3, (50, 10)),
            'y': np.random.randint(0, 3, (50, 1))
        })
        
        train, val, test, meta = load_dataset(str(hard_file))
        assert meta['format'] == 'hard_labels'

    def test_missing_labels_raises(self):
        """Test that missing labels raises error for tensor input"""
        data = torch.rand(50, 10)
        
        with pytest.raises(ValueError, match="labels must be provided"):
            load_dataset(data)

    def test_split_sizes(self):
        """Test that splits have expected relative sizes"""
        data = torch.randint(0, 3, (100, 10))
        labels = torch.randint(0, 3, (100,))
        
        train, val, test, meta = load_dataset(
            data, labels, 
            val_split=0.1, 
            test_split=0.2
        )
        
        # Count samples in each loader
        train_count = sum(len(b[0]) for b in train)
        val_count = sum(len(b[0]) for b in val)
        test_count = sum(len(b[0]) for b in test)
        
        total = train_count + val_count + test_count
        assert total == 100
        assert test_count == 20  # 20% test
        # Remaining 80 split 90/10 = 72 train, 8 val


class TestIntegration:
    """Integration tests with real dataset files."""
    
    def test_load_real_dataset(self):
        """Test loading a real dataset from the datasets folder"""
        dataset_path = Path("/home/dsi/maymona3/rbm_python/datasets/tree3k.mat")
        
        if not dataset_path.exists():
            pytest.skip("Real dataset not available")
        
        train, val, test, meta = load_dataset(str(dataset_path))
        
        assert meta['n_samples'] > 0
        assert meta['n_classifiers'] > 0
        assert meta['n_classes'] > 0
        assert meta['format'] == 'hard_labels'

    def test_load_soft_dataset(self):
        """Test loading a soft labels dataset"""
        dataset_path = Path("/home/dsi/maymona3/rbm_python/datasets/cifar100_expert_router_pretrained.mat")
        
        if not dataset_path.exists():
            pytest.skip("Soft labels dataset not available")
        
        # Check if it's actually soft labels
        data = scipy.io.loadmat(str(dataset_path))
        if len(data['f'].shape) != 3:
            pytest.skip("Dataset is not soft labels")
        
        train, val, test, meta = load_dataset(str(dataset_path))
        assert meta['format'] == 'soft_labels'


class TestBackwardCompatibility:
    """Tests for backward compatibility with old code."""
    
    def test_tensor_dataset_interface(self):
        """Test that TensorDatasetWithShape works like TensorDataset"""
        data = torch.rand(50, 10)
        labels = torch.randint(0, 2, (50,))
        dataset = TensorDatasetWithShape(data, labels)
        
        # Should support indexing
        x, y = dataset[0]
        assert x.shape == (10,)
        
        # Should support len
        assert len(dataset) == 50
        
        # Should support iteration via DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (10, 10)

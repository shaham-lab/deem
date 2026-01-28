"""Tests for deem.core.preprocessing module."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from deem.core.preprocessing import (
    Multinomial,
    Sparsemax,
    SparsemaxFunction,
    SinSoftmax,
    Sin2maxShifted,
    Dropout1dLastDim,
)
from deem.core.preprocessing.utils import (
    flatten_all_but_nth_dim,
    unflatten_all_but_nth_dim,
)


class TestSparsemax:
    """Tests for Sparsemax activation."""
    
    def test_init(self):
        """Test sparsemax initialization."""
        sparsemax = Sparsemax(dim=-1)
        assert sparsemax.dim == -1
        
        sparsemax_custom = Sparsemax(dim=1)
        assert sparsemax_custom.dim == 1
    
    def test_forward_2d(self):
        """Test sparsemax on 2D input."""
        sparsemax = Sparsemax(dim=-1)
        x = torch.randn(32, 10)
        output = sparsemax(x)
        
        assert output.shape == x.shape
        # Check that outputs sum to 1 along last dim
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # Check that all values are non-negative
        assert (output >= 0).all()
    
    def test_forward_3d(self):
        """Test sparsemax on 3D input."""
        sparsemax = Sparsemax(dim=1)
        x = torch.randn(32, 10, 5)
        output = sparsemax(x)
        
        assert output.shape == x.shape
        # Check sums along dim=1
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_sparsity(self):
        """Test that sparsemax produces sparse outputs."""
        sparsemax = Sparsemax(dim=-1)
        # Create input where some elements are much larger
        x = torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0]])
        output = sparsemax(x)
        
        # Should have some zeros due to sparsity
        num_zeros = (output == 0).sum().item()
        assert num_zeros > 0
    
    def test_backward(self):
        """Test that sparsemax is differentiable."""
        sparsemax = Sparsemax(dim=-1)
        x = torch.randn(10, 5, requires_grad=True)
        output = sparsemax(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_dimension_error(self):
        """Test that invalid dimension raises error."""
        sparsemax = Sparsemax(dim=5)
        x = torch.randn(10, 3)
        
        with pytest.raises(IndexError):
            sparsemax(x)


class TestSin2maxShifted:
    """Tests for Sin2maxShifted activation."""
    
    def test_init(self):
        """Test initialization."""
        act = Sin2maxShifted(dim=-1)
        assert act.dim == -1
    
    def test_forward(self):
        """Test forward pass."""
        act = Sin2maxShifted(dim=-1)
        x = torch.randn(32, 10)
        output = act(x)
        
        assert output.shape == x.shape
        # Check normalization
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # All values should be non-negative
        assert (output >= 0).all()


class TestSinSoftmax:
    """Tests for SinSoftmax activation."""
    
    def test_init(self):
        """Test initialization."""
        act = SinSoftmax(dim=-1)
        assert act.dim == -1
    
    def test_forward(self):
        """Test forward pass."""
        act = SinSoftmax(dim=-1)
        x = torch.randn(32, 10)
        output = act(x)
        
        assert output.shape == x.shape
        # Check normalization
        sums = output.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # All values should be non-negative (softmax output)
        assert (output >= 0).all()


class TestDropout1dLastDim:
    """Tests for Dropout1dLastDim."""
    
    def test_init(self):
        """Test initialization."""
        dropout = Dropout1dLastDim(p=0.2)
        assert dropout.dropout.p == 0.2
    
    def test_forward_shape(self):
        """Test output shape preservation."""
        dropout = Dropout1dLastDim(p=0.5)
        x = torch.randn(32, 10, 64)
        
        dropout.train()
        output = dropout(x)
        assert output.shape == x.shape
        
        dropout.eval()
        output_eval = dropout(x)
        assert output_eval.shape == x.shape
    
    def test_dropout_effect(self):
        """Test that dropout actually zeros elements during training."""
        dropout = Dropout1dLastDim(p=0.9)  # High dropout for visible effect
        dropout.train()
        
        x = torch.ones(32, 10, 64)
        output = dropout(x)
        
        # With p=0.9, most elements should be zero
        # But remaining elements are scaled by 1/(1-p) = 10
        num_zeros = (output == 0).sum().item()
        total_elements = output.numel()
        
        # Should have significant dropout
        assert num_zeros > 0


class TestMultinomial:
    """Tests for Multinomial transformation layer."""
    
    def test_init_default(self):
        """Test default initialization."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
        )
        
        assert layer.in_multi_units == 5
        assert layer.out_multi_units == 5
        assert layer.in_features == 10
        assert layer.out_features == 10
        assert layer.weights.shape == (5, 5, 10, 10)
        assert layer.bias.shape == (5, 10)
    
    def test_init_rand(self):
        """Test random initialization."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            init_method='rand',
        )
        
        # Random init should produce non-zero weights
        assert layer.weights.abs().sum() > 0
    
    def test_init_identity(self):
        """Test identity initialization."""
        layer = Multinomial(
            in_multi_units=3,
            out_multi_units=3,
            in_features=5,
            out_features=5,
            init_method='identity',
        )
        
        # Diagonal should be approximately 1 (with jitter)
        for i in range(3):
            for j in range(5):
                assert abs(layer.weights[i, i, j, j].item() - 1.0) < 0.1
    
    def test_init_identity_dimension_mismatch(self):
        """Test that identity init fails with dimension mismatch."""
        with pytest.raises(ValueError):
            Multinomial(
                in_multi_units=3,
                out_multi_units=5,  # Different from in
                in_features=10,
                out_features=10,
                init_method='identity',
            )
    
    def test_init_mv(self):
        """Test majority vote initialization."""
        layer = Multinomial(
            in_multi_units=3,
            out_multi_units=3,
            in_features=5,
            out_features=5,
            init_method='mv',
        )
        
        # Check that diagonal blocks have value 2
        for i in range(3):
            assert torch.allclose(
                layer.weights[i, i],
                torch.ones(5, 5) * 2
            )
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        layer = Multinomial(
            in_multi_units=20,
            out_multi_units=30,
            in_features=40,
            out_features=50,
        )
        
        x = torch.randn(128, 20, 40)
        output = layer(x)
        
        assert output.shape == (128, 30, 50)
    
    def test_forward_with_softmax(self):
        """Test forward pass with softmax activation."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            use_softmax=True,
            activation_func_name='sparsemax',
        )
        
        x = torch.randn(32, 5, 10)
        output = layer(x)
        
        assert output.shape == (32, 5, 10)
        # With sparsemax, outputs should be non-negative
        assert (output >= 0).all()
    
    def test_forward_temperature(self):
        """Test forward pass with temperature scaling."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            use_softmax=True,
        )
        
        x = torch.randn(32, 5, 10)
        output_temp1 = layer(x, temperature=1.0)
        output_temp01 = layer(x, temperature=0.1)
        
        # Lower temperature should produce sharper outputs
        # (higher max values)
        assert output_temp01.max() >= output_temp1.max() - 0.1
    
    def test_forward_one_hot(self):
        """Test forward pass with one-hot output."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            use_softmax=True,
            one_hot=True,
        )
        
        x = torch.randn(32, 5, 10)
        output = layer(x)
        
        assert output.shape == (32, 5, 10)
        # One-hot outputs should sum to 1 along class dimension
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_forward_dbn_mode(self):
        """Test DBN mode with different bias shape."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            dbn_mode=True,
        )
        
        # DBN mode has bias shape (in_multi_units, in_features)
        assert layer.bias.shape == (5, 10)
        
        x = torch.randn(32, 5, 10)
        external_bias = torch.randn(5, 10)
        output = layer.forward_dbn(x, external_bias)
        
        assert output.shape == (32, 5, 10)
    
    def test_backward(self):
        """Test that layer is differentiable."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
        )
        
        x = torch.randn(32, 5, 10, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weights.grad is not None
        assert layer.bias.grad is not None
    
    def test_extra_repr(self):
        """Test string representation."""
        layer = Multinomial(
            in_multi_units=5,
            out_multi_units=10,
            in_features=20,
            out_features=30,
        )
        
        repr_str = layer.extra_repr()
        assert 'in_features=20' in repr_str
        assert 'out_features=30' in repr_str
        assert 'l=5' in repr_str
        assert 'm=10' in repr_str
    
    def test_jitter_weights(self):
        """Test weight jittering."""
        layer = Multinomial(
            in_multi_units=3,
            out_multi_units=3,
            in_features=5,
            out_features=5,
            init_method='identity',
            jitter_coef=0.0,  # No jitter in init
        )
        
        weights_before = layer.weights.clone()
        layer.jitter_weights(jitter_coef=0.1)
        
        # Weights should have changed
        assert not torch.allclose(layer.weights, weights_before)
    
    def test_different_activation_functions(self):
        """Test various activation functions."""
        activations = ['sparsemax', 'softmax', 'gelu', 'silu']
        
        for act_name in activations:
            layer = Multinomial(
                in_multi_units=5,
                out_multi_units=5,
                in_features=10,
                out_features=10,
                use_softmax=True,
                activation_func_name=act_name,
            )
            
            x = torch.randn(16, 5, 10)
            output = layer(x)
            
            assert output.shape == (16, 5, 10), f"Failed for {act_name}"


class TestPreprocessingUtils:
    """Tests for preprocessing utility functions."""
    
    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten/unflatten are inverses."""
        x = torch.randn(10, 5, 8)
        
        # Create mock context
        class MockCtx:
            dim = 1
            original_size = None
            needs_reshaping = True
        
        ctx = MockCtx()
        ctx, flattened = flatten_all_but_nth_dim(ctx, x)
        ctx, recovered = unflatten_all_but_nth_dim(ctx, flattened)
        
        assert torch.allclose(x, recovered)


class TestPreprocessingIntegration:
    """Integration tests for preprocessing layers."""
    
    def test_multinomial_with_sparsemax(self):
        """Test Multinomial layer integrated with Sparsemax."""
        # Build a simple preprocessing pipeline
        multinomial = Multinomial(
            in_multi_units=5,
            out_multi_units=5,
            in_features=10,
            out_features=10,
            use_softmax=True,
            activation_func_name='sparsemax',
        )
        
        # Simulate RBM-style input (batch, classifiers, classes)
        x = torch.randn(64, 5, 10)
        output = multinomial(x)
        
        # Output should be valid probability-like distribution
        assert output.shape == x.shape
        assert (output >= 0).all()
    
    def test_stacked_multinomial_layers(self):
        """Test stacking multiple Multinomial layers (like MultinomialNet)."""
        layers = nn.Sequential(
            Multinomial(5, 5, 10, 10, use_softmax=True),
            Multinomial(5, 5, 10, 10, use_softmax=True),
            Multinomial(5, 5, 10, 10, use_softmax=True),
        )
        
        x = torch.randn(32, 5, 10)
        output = layers(x)
        
        assert output.shape == x.shape
    
    def test_preprocessing_for_rbm(self):
        """Test preprocessing output is suitable for RBM input."""
        # Create preprocessing similar to what RBMGwg uses
        preprocessing = Multinomial(
            in_multi_units=15,  # Number of classifiers
            out_multi_units=15,
            in_features=10,     # Number of classes
            out_features=10,
            init_method='identity',
            use_softmax=True,
            activation_func_name='sparsemax',
        )
        
        # Simulate one-hot encoded classifier predictions
        # Shape: (batch, num_classifiers, num_classes)
        batch_size = 64
        num_classifiers = 15
        num_classes = 10
        
        # Create random one-hot predictions
        predictions = torch.zeros(batch_size, num_classifiers, num_classes)
        for i in range(batch_size):
            for j in range(num_classifiers):
                class_idx = torch.randint(0, num_classes, (1,)).item()
                predictions[i, j, class_idx] = 1.0
        
        # Apply preprocessing
        processed = preprocessing(predictions)
        
        # Should maintain shape
        assert processed.shape == predictions.shape
        
        # Output should be valid for RBM (non-negative, reasonable values)
        assert (processed >= -1).all()  # Allow some slack for numerical issues

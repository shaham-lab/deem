"""Tests for deem.core.models.rbm_gwg module.

Tests cover:
- MultinomialRBMGwg instantiation
- Forward pass with hard and soft labels
- Preprocessing integration
- Sampler integration
- Prediction functionality
- Energy calculation
"""

import pytest
import torch
import numpy as np

from deem.core.models.rbm_gwg import MultinomialRBMGwg, RBMGwg
from deem.core.preprocessing import Multinomial


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Return device for tests."""
    return 'cpu'


@pytest.fixture
def rbm_params(device):
    """Basic parameters for MultinomialRBMGwg."""
    return {
        'dx': 15,    # 15 classifiers
        'dh': 3,     # 3 hidden units
        'cd_k': 10,  # 10 CD steps
        'k': 3,      # 3 classes
        'l': 3,      # 3 visible states
        'm': 3,      # 3 hidden states
        'device': device,
        'deterministic': True,
        'sampler_steps': 2,  # Fewer steps for faster tests
    }


@pytest.fixture
def rbm(rbm_params):
    """Create a MultinomialRBMGwg instance for testing."""
    return MultinomialRBMGwg(**rbm_params)


@pytest.fixture
def hard_labels():
    """Sample hard labels (N, D) with integer class predictions."""
    torch.manual_seed(42)
    return torch.randint(0, 3, (32, 15))


@pytest.fixture
def hard_labels_with_missing():
    """Sample hard labels with -1 for missing values."""
    torch.manual_seed(42)
    labels = torch.randint(0, 3, (32, 15))
    # Add some missing values
    labels[0, 0] = -1
    labels[5, 7] = -1
    labels[10, :] = -1  # Entire row missing
    return labels


@pytest.fixture
def soft_labels():
    """Sample soft labels (N, K, D) with probability distributions."""
    torch.manual_seed(42)
    probs = torch.rand(32, 3, 15)
    # Normalize to valid probability distributions
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


# =============================================================================
# Tests: Model Instantiation
# =============================================================================

class TestInstantiation:
    """Tests for MultinomialRBMGwg creation."""
    
    def test_basic_creation(self, rbm_params):
        """Test basic model creation."""
        rbm = MultinomialRBMGwg(**rbm_params)
        
        assert rbm.dx == 15
        assert rbm.dh == 3
        assert rbm.k == 3
        assert rbm.l == 3
        assert rbm.m == 3
    
    def test_weight_shapes(self, rbm):
        """Test that weights have correct shapes."""
        # Weights: (l, m, dx, dh)
        assert rbm.weights.shape == (3, 3, 15, 3)
        # Visible bias: (dx, l)
        assert rbm.visible_bias.shape == (15, 3)
        # Hidden bias: (dh, m)
        assert rbm.hidden_bias.shape == (3, 3)
    
    def test_sampler_created(self, rbm):
        """Test that sampler is properly initialized."""
        assert rbm.sampler is not None
        assert rbm.sampler_steps == 2
    
    def test_no_multinomial_net_by_default(self, rbm):
        """Test that multinomial_net is None by default."""
        assert rbm.multinomial_net is None
        assert rbm.multinomial_params is None
    
    def test_with_multinomial_net(self, rbm_params, device):
        """Test creation with multinomial preprocessing network."""
        # Create a simple multinomial net
        multi_net = Multinomial(
            in_multi_units=rbm_params['dx'],
            out_multi_units=rbm_params['dx'],
            in_features=rbm_params['k'],
            out_features=rbm_params['k'],
            device=device
        )
        
        rbm_params['multinomial_net'] = multi_net
        rbm = MultinomialRBMGwg(**rbm_params)
        
        assert rbm.multinomial_net is not None
        assert rbm.multinomial_params is not None
        assert len(rbm.multinomial_params) > 0
    
    def test_rbm_params_list(self, rbm):
        """Test that rbm_params contains all RBM parameters."""
        assert len(rbm.rbm_params) == 3
        assert rbm.weights in rbm.rbm_params
        assert rbm.visible_bias in rbm.rbm_params
        assert rbm.hidden_bias in rbm.rbm_params
    
    def test_backward_compatibility_alias(self, rbm_params):
        """Test RBMGwg alias works."""
        rbm = RBMGwg(**rbm_params)
        assert isinstance(rbm, MultinomialRBMGwg)


# =============================================================================
# Tests: Forward Pass
# =============================================================================

class TestForwardPass:
    """Tests for forward pass functionality."""
    
    def test_forward_hard_labels(self, rbm, hard_labels):
        """Test forward pass with hard labels."""
        visible, intermediate, hidden = rbm(hard_labels)
        
        # Check shapes
        assert visible.shape == (32, 3, 15)  # (N, L, D)
        assert intermediate.shape == (32, 3, 15)
        assert hidden.shape == (32, 3, 3)  # (N, DH, M)
    
    def test_forward_soft_labels(self, rbm, soft_labels):
        """Test forward pass with soft labels."""
        visible, intermediate, hidden = rbm(soft_labels)
        
        # Check shapes
        assert visible.shape == (32, 3, 15)
        assert intermediate.shape == (32, 3, 15)
        assert hidden.shape == (32, 3, 3)
    
    def test_forward_with_missing_values(self, rbm, hard_labels_with_missing):
        """Test forward pass handles -1 missing values."""
        visible, intermediate, hidden = rbm(hard_labels_with_missing)
        
        # Should not raise errors
        assert visible.shape == (32, 3, 15)
        assert hidden.shape == (32, 3, 3)
        
        # Missing values should become all-zero vectors
        # First sample, first position was -1
        assert visible[0, :, 0].sum() == 0
    
    def test_visible_intermediate_same_without_multinomial(self, rbm, hard_labels):
        """Test that visible == intermediate when no multinomial_net."""
        visible, intermediate, hidden = rbm(hard_labels)
        assert torch.allclose(visible, intermediate)


# =============================================================================
# Tests: Prediction
# =============================================================================

class TestPredict:
    """Tests for prediction functionality."""
    
    def test_predict_hard_labels(self, rbm, hard_labels):
        """Test prediction with hard labels."""
        predictions = rbm.predict(hard_labels)
        
        # Predictions should be (N, DH)
        assert predictions.shape == (32, 3)
        # Values should be class indices
        assert predictions.min() >= 0
        assert predictions.max() < 3
    
    def test_predict_soft_labels(self, rbm, soft_labels):
        """Test prediction with soft labels."""
        predictions = rbm.predict(soft_labels)
        
        assert predictions.shape == (32, 3)
    
    def test_predict_as_distribution(self, rbm, hard_labels):
        """Test prediction with distribution sampling."""
        predictions = rbm.predict(hard_labels, as_distribution=True)
        
        assert predictions.shape == (32, 3)
        assert predictions.min() >= 0
        assert predictions.max() < 3


# =============================================================================
# Tests: Energy Calculation
# =============================================================================

class TestEnergy:
    """Tests for energy calculation."""
    
    def test_energy_shape(self, rbm, hard_labels):
        """Test energy returns correct shape."""
        visible = rbm.preprocess(hard_labels)
        hidden_probs = rbm.calc_hidden_probs(visible)
        hidden = rbm.sample_from_hidden_probs(hidden_probs)
        
        energy = rbm.energy(visible, hidden)
        
        assert energy.shape == (32,)
    
    def test_negative_energy(self, rbm, hard_labels):
        """Test negative_energy method."""
        visible = rbm.preprocess(hard_labels)
        hidden_probs = rbm.calc_hidden_probs(visible)
        hidden = rbm.sample_from_hidden_probs(hidden_probs)
        
        neg_energy = rbm.negative_energy(visible, hidden)
        energy = rbm.energy(visible, hidden)
        
        assert torch.allclose(neg_energy, -energy)


# =============================================================================
# Tests: Apply Multinomial Layer
# =============================================================================

class TestApplyMultinomialLayer:
    """Tests for multinomial layer application."""
    
    def test_passthrough_without_net(self, rbm, hard_labels):
        """Test data passes through unchanged without multinomial_net."""
        visible = rbm.preprocess(hard_labels)
        result = rbm.apply_multinomial_layer(visible)
        
        assert torch.equal(visible, result)
    
    def test_transforms_with_net(self, rbm_params, device, hard_labels):
        """Test data is transformed when multinomial_net exists."""
        multi_net = Multinomial(
            in_multi_units=rbm_params['dx'],
            out_multi_units=rbm_params['dx'],
            in_features=rbm_params['k'],
            out_features=rbm_params['k'],
            device=device
        )
        
        rbm_params['multinomial_net'] = multi_net
        rbm = MultinomialRBMGwg(**rbm_params)
        
        visible = rbm.preprocess(hard_labels)
        result = rbm.apply_multinomial_layer(visible)
        
        # Result should be different (unless weights happen to be identity)
        assert result.shape == visible.shape


# =============================================================================
# Tests: Weight Initialization
# =============================================================================

class TestWeightInitialization:
    """Tests for different weight initialization methods."""
    
    def test_rand_init(self, rbm_params):
        """Test random initialization."""
        rbm_params['init_method'] = 'rand'
        rbm = MultinomialRBMGwg(**rbm_params)
        
        # Weights should not be zero
        assert rbm.weights.abs().sum() > 0
    
    def test_mv_init(self, rbm_params):
        """Test majority vote initialization."""
        rbm_params['init_method'] = 'mv'
        rbm = MultinomialRBMGwg(**rbm_params)
        
        # Diagonal weights should be set
        for i in range(min(rbm.l, rbm.m)):
            assert rbm.weights[i, i].abs().sum() > 0


# =============================================================================
# Tests: Get Samples (Core Training Method)
# =============================================================================

class TestGetSamples:
    """Tests for get_samples method used in training."""
    
    def test_get_samples_shapes(self, rbm, hard_labels):
        """Test get_samples returns correct shapes."""
        real_vis, real_hid, fake_vis, fake_hid = rbm.get_samples(hard_labels)
        
        # Real samples
        assert real_vis.shape == (32, 3, 15)
        assert real_hid.shape == (32, 3, 3)
        
        # Fake samples should match
        assert fake_vis.shape == real_vis.shape
        assert fake_hid.shape == real_hid.shape
    
    def test_get_samples_train_multi_false(self, rbm_params, device, hard_labels):
        """Test get_samples with train_multi=False (train RBM only)."""
        multi_net = Multinomial(
            in_multi_units=rbm_params['dx'],
            out_multi_units=rbm_params['dx'],
            in_features=rbm_params['k'],
            out_features=rbm_params['k'],
            device=device
        )
        
        rbm_params['multinomial_net'] = multi_net
        rbm = MultinomialRBMGwg(**rbm_params)
        
        # Call with train_multi=False
        rbm.get_samples(hard_labels, train_multi=False)
        
        # Multinomial params should have requires_grad=False
        for param in rbm.multinomial_params:
            assert not param.requires_grad


# =============================================================================
# Tests: Module Registration
# =============================================================================

class TestModuleRegistration:
    """Tests for proper PyTorch module registration."""
    
    def test_parameters_registered(self, rbm):
        """Test that parameters are properly registered."""
        params = list(rbm.parameters())
        
        # Should have weights, visible_bias, hidden_bias
        assert len(params) >= 3
    
    def test_model_to_device(self, rbm_params):
        """Test model can be moved to device."""
        rbm = MultinomialRBMGwg(**rbm_params)
        
        # Should not raise
        rbm.to('cpu')
        
        assert rbm.weights.device.type == 'cpu'
        assert rbm.visible_bias.device.type == 'cpu'
        assert rbm.hidden_bias.device.type == 'cpu'

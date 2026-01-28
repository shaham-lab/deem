"""Tests for deem.core.models.base module.

Tests cover:
- RBM abstract class interface
- MultinomialRBM preprocess() method for hard/soft labels
- Weight initialization methods
- Sampling and probability calculations
"""

import pytest
import torch
import numpy as np

from deem.core.models.base import RBM, MultinomialRBM, one_hot_encode


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Return device for tests."""
    return 'cpu'


@pytest.fixture
def mock_rbm_params(device):
    """Basic parameters for MultinomialRBM."""
    return {
        'dx': 15,    # 15 classifiers
        'dh': 1,     # 1 hidden unit
        'cd_k': 10,  # 10 CD steps
        'k': 3,      # 3 classes
        'l': 3,      # 3 visible states
        'm': 3,      # 3 hidden states
        'device': device,
        'deterministic': True,
    }


@pytest.fixture
def mock_rbm(mock_rbm_params):
    """Create a MultinomialRBM instance for testing."""
    return MultinomialRBM(**mock_rbm_params)


# =============================================================================
# Tests: one_hot_encode function
# =============================================================================

class TestOneHotEncode:
    """Tests for one_hot_encode utility function."""
    
    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        tensor = torch.tensor([0, 1, 2])
        result = one_hot_encode(tensor, num_classes=3)
        expected = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert torch.equal(result, expected)
    
    def test_2d_encoding(self):
        """Test one-hot encoding of 2D tensor."""
        tensor = torch.tensor([[0, 1], [2, 0]])
        result = one_hot_encode(tensor, num_classes=3)
        assert result.shape == (2, 2, 3)
        # First sample: [0, 1] -> [[1,0,0], [0,1,0]]
        assert torch.equal(result[0, 0], torch.tensor([1, 0, 0]))
        assert torch.equal(result[0, 1], torch.tensor([0, 1, 0]))
    
    def test_relaxed_negative_values(self):
        """Test relaxed mode handles -1 (missing values)."""
        tensor = torch.tensor([0, -1, 2])
        result = one_hot_encode(tensor, num_classes=3, relaxed=True)
        # -1 should become all zeros
        assert torch.equal(result[0], torch.tensor([1, 0, 0]))
        assert torch.equal(result[1], torch.tensor([0, 0, 0]))  # Missing value
        assert torch.equal(result[2], torch.tensor([0, 0, 1]))
    
    def test_relaxed_out_of_range(self):
        """Test relaxed mode handles out-of-range values."""
        tensor = torch.tensor([0, 5, 2])
        result = one_hot_encode(tensor, num_classes=3, relaxed=True)
        # 5 is out of range, should become all zeros
        assert torch.equal(result[1], torch.tensor([0, 0, 0]))
    
    def test_strict_raises_on_invalid(self):
        """Test strict mode raises on invalid values."""
        tensor = torch.tensor([0, -1, 2])
        with pytest.raises(ValueError, match="Input values must be between"):
            one_hot_encode(tensor, num_classes=3, relaxed=False)


# =============================================================================
# Tests: RBM abstract class
# =============================================================================

class TestRBMAbstract:
    """Tests for RBM abstract base class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that RBM cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RBM(dx=10, dh=1, cd_k=10, device='cpu')
    
    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""
        class IncompleteRBM(RBM):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRBM(dx=10, dh=1, cd_k=10, device='cpu')


# =============================================================================
# Tests: MultinomialRBM initialization
# =============================================================================

class TestMultinomialRBMInit:
    """Tests for MultinomialRBM initialization."""
    
    def test_default_init(self, mock_rbm_params):
        """Test default random initialization."""
        rbm = MultinomialRBM(**mock_rbm_params)
        
        assert rbm.weights.shape == (3, 3, 15, 1)  # (l, m, dx, dh)
        assert rbm.visible_bias.shape == (15, 3)    # (dx, l)
        assert rbm.hidden_bias.shape == (1, 3)      # (dh, m)
    
    def test_mv_init(self, mock_rbm_params):
        """Test majority vote initialization."""
        mock_rbm_params['init_method'] = 'mv'
        rbm = MultinomialRBM(**mock_rbm_params)
        
        # Diagonal should be set to irbm_mv_impact
        for i in range(3):
            assert torch.allclose(
                rbm.weights[i, i],
                torch.ones(15, 1, device=rbm.device) * rbm.irbm_mv_impact
            )
    
    def test_parameters_are_registered(self, mock_rbm):
        """Test that weights and biases are registered as parameters."""
        param_names = [name for name, _ in mock_rbm.named_parameters()]
        assert 'weights' in param_names
        assert 'visible_bias' in param_names
        assert 'hidden_bias' in param_names
    
    def test_device_assignment(self, mock_rbm_params):
        """Test device is correctly assigned."""
        rbm = MultinomialRBM(**mock_rbm_params)
        assert str(rbm.weights.device) == 'cpu'


# =============================================================================
# Tests: preprocess() method
# =============================================================================

class TestPreprocess:
    """Tests for MultinomialRBM.preprocess() method."""
    
    def test_hard_labels_to_onehot(self, mock_rbm):
        """Test conversion of 2D hard labels to 3D one-hot."""
        # Hard labels: (N, D) = (10, 15)
        hard_labels = torch.randint(0, 3, (10, 15))
        
        result = mock_rbm.preprocess(hard_labels)
        
        # Should be (N, K, D) = (10, 3, 15)
        assert result.shape == (10, 3, 15)
        # Should be one-hot along K dimension
        assert torch.allclose(result.sum(dim=1), torch.ones(10, 15))
    
    def test_soft_labels_passthrough(self, mock_rbm):
        """Test that 3D soft labels pass through unchanged."""
        # Soft labels: (N, K, D) = (10, 3, 15)
        soft_labels = torch.rand(10, 3, 15)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)  # Normalize
        
        result = mock_rbm.preprocess(soft_labels)
        
        assert result.shape == (10, 3, 15)
        assert torch.equal(result, soft_labels)
    
    def test_hard_labels_with_missing(self, mock_rbm):
        """Test handling of -1 (missing values) in hard labels."""
        hard_labels = torch.randint(0, 3, (10, 15))
        hard_labels[0, 0] = -1  # Missing value
        hard_labels[5, 7] = -1  # Another missing value
        
        result = mock_rbm.preprocess(hard_labels)
        
        assert result.shape == (10, 3, 15)
        # Missing values should be all zeros
        assert torch.equal(result[0, :, 0], torch.zeros(3))
        assert torch.equal(result[5, :, 7], torch.zeros(3))
    
    def test_preprocess_preserves_valid_values(self, mock_rbm):
        """Test that valid values are correctly encoded."""
        hard_labels = torch.tensor([[0, 1, 2]])  # (1, 3)
        
        result = mock_rbm.preprocess(hard_labels)
        
        # Check each position
        assert torch.equal(result[0, :, 0], torch.tensor([1., 0., 0.]))  # Class 0
        assert torch.equal(result[0, :, 1], torch.tensor([0., 1., 0.]))  # Class 1
        assert torch.equal(result[0, :, 2], torch.tensor([0., 0., 1.]))  # Class 2


# =============================================================================
# Tests: Probability calculations
# =============================================================================

class TestProbabilityCalculations:
    """Tests for probability calculation methods."""
    
    def test_calc_hidden_probs_shape(self, mock_rbm):
        """Test calc_hidden_probs returns correct shape."""
        visible = torch.rand(10, 3, 15)  # (N, L, D)
        
        hidden_probs = mock_rbm.calc_hidden_probs(visible)
        
        assert hidden_probs.shape == (10, 1, 3)  # (N, H, M)
    
    def test_calc_hidden_probs_normalized(self, mock_rbm):
        """Test calc_hidden_probs returns valid probabilities."""
        visible = torch.rand(10, 3, 15)
        
        hidden_probs = mock_rbm.calc_hidden_probs(visible)
        
        # Should sum to 1 along last dimension (M)
        sums = hidden_probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    def test_calc_visible_probs_shape(self, mock_rbm):
        """Test calc_visible_probs returns correct shape."""
        hidden = torch.rand(10, 1, 3)  # (N, H, M)
        
        visible_probs = mock_rbm.calc_visible_probs(hidden)
        
        assert visible_probs.shape == (10, 3, 15)  # (N, L, D)
    
    def test_calc_visible_probs_normalized(self, mock_rbm):
        """Test calc_visible_probs returns valid probabilities."""
        hidden = torch.rand(10, 1, 3)
        
        visible_probs = mock_rbm.calc_visible_probs(hidden)
        
        # Should sum to 1 along dimension 1 (L)
        sums = visible_probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


# =============================================================================
# Tests: Sampling
# =============================================================================

class TestSampling:
    """Tests for sampling methods."""
    
    def test_sample_hidden_deterministic(self, mock_rbm):
        """Test deterministic mode returns probabilities."""
        mock_rbm.deterministic = True
        hidden_probs = torch.rand(10, 1, 3)
        hidden_probs = hidden_probs / hidden_probs.sum(dim=2, keepdim=True)
        
        result = mock_rbm.sample_from_hidden_probs(hidden_probs)
        
        assert torch.equal(result, hidden_probs)
    
    def test_sample_hidden_stochastic(self, mock_rbm):
        """Test stochastic mode returns one-hot samples."""
        mock_rbm.deterministic = False
        mock_rbm.as_distribution = True
        hidden_probs = torch.rand(10, 1, 3)
        hidden_probs = hidden_probs / hidden_probs.sum(dim=2, keepdim=True)
        
        result = mock_rbm.sample_from_hidden_probs(hidden_probs)
        
        # Should be one-hot
        assert torch.allclose(result.sum(dim=2), torch.ones(10, 1))
        # All values should be 0 or 1
        assert torch.all((result == 0) | (result == 1))
    
    def test_sample_visible_deterministic(self, mock_rbm):
        """Test deterministic mode returns probabilities."""
        mock_rbm.deterministic = True
        visible_probs = torch.rand(10, 3, 15)
        visible_probs = visible_probs / visible_probs.sum(dim=1, keepdim=True)
        
        result = mock_rbm.sample_from_visible_probs(visible_probs)
        
        assert torch.equal(result, visible_probs)


# =============================================================================
# Tests: Energy calculation
# =============================================================================

class TestEnergy:
    """Tests for energy calculation."""
    
    def test_energy_shape(self, mock_rbm):
        """Test energy returns correct shape."""
        visible = torch.rand(10, 3, 15)
        hidden = torch.rand(10, 1, 3)
        
        energy = mock_rbm.energy(visible, hidden)
        
        assert energy.shape == (10,)
    
    def test_negative_energy(self, mock_rbm):
        """Test negative_energy is negation of energy."""
        visible = torch.rand(10, 3, 15)
        hidden = torch.rand(10, 1, 3)
        
        e = mock_rbm.energy(visible, hidden)
        neg_e = mock_rbm.negative_energy(visible, hidden)
        
        assert torch.allclose(neg_e, -e)


# =============================================================================
# Tests: Contrastive Divergence
# =============================================================================

class TestContrastiveDivergence:
    """Tests for contrastive divergence."""
    
    def test_cd_returns_four_tensors(self, mock_rbm):
        """Test CD returns positive and negative samples."""
        hard_labels = torch.randint(0, 3, (10, 15))
        
        pos_v, pos_h, neg_v, neg_h = mock_rbm.contrastive_divergence(hard_labels)
        
        assert pos_v.shape == (10, 3, 15)  # Preprocessed visible
        assert pos_h.shape == (10, 1, 3)   # Hidden
        assert neg_v.shape == (10, 3, 15)  # Reconstructed visible
        assert neg_h.shape == (10, 1, 3)   # Reconstructed hidden
    
    def test_cd_preprocesses_input(self, mock_rbm):
        """Test CD correctly preprocesses hard labels."""
        hard_labels = torch.randint(0, 3, (10, 15))
        
        pos_v, _, _, _ = mock_rbm.contrastive_divergence(hard_labels)
        
        # pos_v should be one-hot encoded
        assert torch.allclose(pos_v.sum(dim=1), torch.ones(10, 15))


# =============================================================================
# Tests: Predict
# =============================================================================

class TestPredict:
    """Tests for predict method."""
    
    def test_predict_hard_labels(self, mock_rbm):
        """Test predict with hard labels."""
        hard_labels = torch.randint(0, 3, (10, 15))
        
        predictions = mock_rbm.predict(hard_labels)
        
        assert predictions.shape == (10, 1)  # (N, H)
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < 3)
    
    def test_predict_soft_labels(self, mock_rbm):
        """Test predict with soft labels."""
        soft_labels = torch.rand(10, 3, 15)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        
        predictions = mock_rbm.predict(soft_labels)
        
        assert predictions.shape == (10, 1)


# =============================================================================
# Tests: Weight copying
# =============================================================================

class TestWeightCopying:
    """Tests for weight copying utility."""
    
    def test_copy_weights(self, mock_rbm_params):
        """Test copying weights between RBMs."""
        source = MultinomialRBM(**mock_rbm_params)
        target = MultinomialRBM(**mock_rbm_params)
        
        # Modify source weights
        source.weights.data.fill_(1.0)
        source.visible_bias.data.fill_(2.0)
        source.hidden_bias.data.fill_(3.0)
        
        RBM.copy_rbm_weights(source, target)
        
        assert torch.equal(target.weights, source.weights)
        assert torch.equal(target.visible_bias, source.visible_bias)
        assert torch.equal(target.hidden_bias, source.hidden_bias)
    
    def test_copy_weights_shape_mismatch(self, mock_rbm_params):
        """Test error on shape mismatch."""
        source = MultinomialRBM(**mock_rbm_params)
        
        mock_rbm_params['dh'] = 5  # Different hidden dim
        target = MultinomialRBM(**mock_rbm_params)
        
        with pytest.raises(ValueError, match="shape"):
            RBM.copy_rbm_weights(source, target)

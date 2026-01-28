"""Tests for deem.core.losses module."""

import pytest
import torch
import torch.nn as nn

from deem.core.losses import EBMLoss, ContrastiveDivergenceLoss


class MockEnergyModel(nn.Module):
    """Mock energy model for testing losses."""

    def __init__(self, return_value: float = 1.0):
        super().__init__()
        self.return_value = return_value
        self.call_count = 0

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Return a simple energy based on input."""
        self.call_count += 1
        batch_size = v.shape[0]
        # Simple energy: sum of all elements
        return v.sum(dim=tuple(range(1, v.dim()))) + h.sum(dim=tuple(range(1, h.dim())))


class TestEBMLoss:
    """Tests for EBMLoss class."""

    def test_init(self):
        """Test EBMLoss initialization."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        assert loss_fn.model is model
        assert loss_fn.with_norm is False

    def test_init_with_norm(self):
        """Test EBMLoss initialization with normalization."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model, with_norm=True)

        assert loss_fn.with_norm is True

    def test_forward_basic(self):
        """Test basic forward pass."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        # Create sample tensors
        batch_size = 16
        v_pos = torch.randn(batch_size, 3, 10)
        h_pos = torch.randn(batch_size, 5)
        v_neg = torch.randn(batch_size, 3, 10)
        h_neg = torch.randn(batch_size, 5)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)

        # Check output shape and type
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_forward_calls_energy_twice(self):
        """Test that forward calls energy for both phases."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10)
        h_pos = torch.randn(8, 5)
        v_neg = torch.randn(8, 10)
        h_neg = torch.randn(8, 5)

        loss_fn(v_pos, h_pos, v_neg, h_neg)

        assert model.call_count == 2

    def test_forward_differentiable(self):
        """Test that loss is differentiable."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10, requires_grad=True)
        h_pos = torch.randn(8, 5, requires_grad=True)
        v_neg = torch.randn(8, 10, requires_grad=True)
        h_neg = torch.randn(8, 5, requires_grad=True)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        loss.backward()

        assert v_pos.grad is not None
        assert h_pos.grad is not None

    def test_loss_sign_convention(self):
        """Test that lower positive energy and higher negative energy decrease loss."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        # Case 1: positive energy < negative energy (good)
        v_pos_small = torch.zeros(8, 10)  # Low energy
        h_pos_small = torch.zeros(8, 5)
        v_neg_large = torch.ones(8, 10)  # High energy
        h_neg_large = torch.ones(8, 5)

        # Case 2: positive energy > negative energy (bad)
        v_pos_large = torch.ones(8, 10)  # High energy
        h_pos_large = torch.ones(8, 5)
        v_neg_small = torch.zeros(8, 10)  # Low energy
        h_neg_small = torch.zeros(8, 5)

        loss_good = loss_fn(v_pos_small, h_pos_small, v_neg_large, h_neg_large)
        loss_bad = loss_fn(v_pos_large, h_pos_large, v_neg_small, h_neg_small)

        # Good case should have lower loss
        assert loss_good < loss_bad

    def test_with_norm_adds_regularization(self):
        """Test that with_norm=True adds energy regularization."""
        model = MockEnergyModel()
        loss_fn_no_norm = EBMLoss(model, with_norm=False)
        loss_fn_with_norm = EBMLoss(model, with_norm=True)

        v_pos = torch.randn(8, 10)
        h_pos = torch.randn(8, 5)
        v_neg = torch.randn(8, 10)
        h_neg = torch.randn(8, 5)

        loss_no_norm = loss_fn_no_norm(v_pos, h_pos, v_neg, h_neg)
        loss_with_norm = loss_fn_with_norm(v_pos, h_pos, v_neg, h_neg)

        # With norm should generally be different (larger due to squared energies)
        # Note: they could be equal in edge cases, but generally won't be
        assert loss_no_norm.item() != loss_with_norm.item()

    def test_batch_size_handling(self):
        """Test that loss works with different batch sizes."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        for batch_size in [1, 8, 32, 128]:
            v_pos = torch.randn(batch_size, 10)
            h_pos = torch.randn(batch_size, 5)
            v_neg = torch.randn(batch_size, 10)
            h_neg = torch.randn(batch_size, 5)

            loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
            assert loss.dim() == 0  # Always scalar

    def test_device_handling_cpu(self):
        """Test that loss works on CPU."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10)
        h_pos = torch.randn(8, 5)
        v_neg = torch.randn(8, 10)
        h_neg = torch.randn(8, 5)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        assert loss.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_handling_cuda(self):
        """Test that loss works on CUDA."""
        model = MockEnergyModel().cuda()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10).cuda()
        h_pos = torch.randn(8, 5).cuda()
        v_neg = torch.randn(8, 10).cuda()
        h_neg = torch.randn(8, 5).cuda()

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        assert loss.device.type == 'cuda'


class TestContrastiveDivergenceLoss:
    """Tests for ContrastiveDivergenceLoss alias."""

    def test_is_ebm_loss(self):
        """Test that ContrastiveDivergenceLoss is EBMLoss."""
        assert ContrastiveDivergenceLoss is EBMLoss

    def test_can_instantiate(self):
        """Test that ContrastiveDivergenceLoss can be instantiated."""
        model = MockEnergyModel()
        loss_fn = ContrastiveDivergenceLoss(model)
        assert isinstance(loss_fn, EBMLoss)


class TestLossWithRealishModel:
    """Integration-like tests with a more realistic model."""

    def test_with_multinomial_like_tensors(self):
        """Test with tensor shapes matching multinomial RBM output."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        # Shape: (batch, k, dx) for visible, (batch, dh) for hidden
        batch_size = 32
        k = 3
        dx = 15
        dh = 1

        v_pos = torch.randn(batch_size, k, dx)
        h_pos = torch.randn(batch_size, dh)
        v_neg = torch.randn(batch_size, k, dx)
        h_neg = torch.randn(batch_size, dh)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss."""

        class TrainableEnergyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 5))

            def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
                # Energy using the weight parameter
                return (v @ self.weight @ h.T).diag()

        model = TrainableEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10)
        h_pos = torch.randn(8, 5)
        v_neg = torch.randn(8, 10)
        h_neg = torch.randn(8, 5)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        loss.backward()

        # Check that model weights have gradients
        assert model.weight.grad is not None
        assert model.weight.grad.shape == model.weight.shape


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_tensors(self):
        """Test with all-zero tensors."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.zeros(8, 10)
        h_pos = torch.zeros(8, 5)
        v_neg = torch.zeros(8, 10)
        h_neg = torch.zeros(8, 5)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)

        assert torch.isfinite(loss)
        assert loss.item() == 0.0  # All energies are 0

    def test_single_sample(self):
        """Test with batch size of 1."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(1, 10)
        h_pos = torch.randn(1, 5)
        v_neg = torch.randn(1, 10)
        h_neg = torch.randn(1, 5)

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        assert torch.isfinite(loss)

    def test_large_values(self):
        """Test with large tensor values."""
        model = MockEnergyModel()
        loss_fn = EBMLoss(model)

        v_pos = torch.randn(8, 10) * 100
        h_pos = torch.randn(8, 5) * 100
        v_neg = torch.randn(8, 10) * 100
        h_neg = torch.randn(8, 5) * 100

        loss = loss_fn(v_pos, h_pos, v_neg, h_neg)
        assert torch.isfinite(loss)

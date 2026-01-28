"""Tests for deem.core.samplers module."""

import pytest
import torch
import torch.nn as nn

from deem.core.samplers import Buffer, GwgSampler, DlpSampler


class TestBuffer:
    """Tests for the Buffer class."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = Buffer(buffer_max_size=100)
        assert buffer.buffer_max_size == 100
        assert len(buffer) == 0
        assert buffer.examples == []

    def test_add_examples_single_batch(self):
        """Test adding a single batch of examples."""
        buffer = Buffer(buffer_max_size=100)
        examples = torch.randn(10, 3, 5)
        buffer.add_examples(examples)
        
        assert len(buffer) == 10

    def test_add_examples_multiple_batches(self):
        """Test adding multiple batches."""
        buffer = Buffer(buffer_max_size=100)
        
        buffer.add_examples(torch.randn(10, 3, 5))
        buffer.add_examples(torch.randn(15, 3, 5))
        
        assert len(buffer) == 25

    def test_add_examples_exceeds_max_size(self):
        """Test that buffer trims when exceeding max size."""
        buffer = Buffer(buffer_max_size=20)
        
        buffer.add_examples(torch.randn(15, 3, 5))
        buffer.add_examples(torch.randn(10, 3, 5))
        
        # Should keep only the most recent 20
        assert len(buffer) == 20

    def test_get_random_examples_all(self):
        """Test getting all examples from buffer."""
        buffer = Buffer(buffer_max_size=100)
        examples = torch.randn(10, 3, 5)
        buffer.add_examples(examples)
        
        result = buffer.get_random_examples()
        
        assert result.shape == (10, 3, 5)

    def test_get_random_examples_subset(self):
        """Test getting a subset of examples."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(50, 3, 5))
        
        result = buffer.get_random_examples(10)
        
        assert result.shape == (10, 3, 5)

    def test_get_random_examples_with_indices(self):
        """Test getting examples with indices."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(50, 3, 5))
        
        result, indices = buffer.get_random_examples(10, with_indices=True)
        
        assert result.shape == (10, 3, 5)
        assert len(indices) == 10
        assert all(0 <= idx < 50 for idx in indices)

    def test_get_random_examples_with_indices_method(self):
        """Test the explicit with_indices method."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(50, 3, 5))
        
        result, indices = buffer.get_random_examples_with_indices(10)
        
        assert result.shape == (10, 3, 5)
        assert len(indices) == 10

    def test_get_random_examples_empty_buffer(self):
        """Test that getting examples from empty buffer raises error."""
        buffer = Buffer(buffer_max_size=100)
        
        with pytest.raises(ValueError, match="Buffer is empty"):
            buffer.get_random_examples()

    def test_modify_by_indices(self):
        """Test modifying examples by indices."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.zeros(10, 3, 5))
        
        new_data = torch.ones(2, 3, 5)
        buffer.modify_by_indices(new_data, [0, 5])
        
        assert torch.all(buffer[0] == 1)
        assert torch.all(buffer[5] == 1)
        assert torch.all(buffer[1] == 0)

    def test_modify_by_indices_mismatched_size(self):
        """Test modify raises error on size mismatch."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(10, 3, 5))
        
        with pytest.raises(ValueError, match="must match"):
            buffer.modify_by_indices(torch.randn(3, 3, 5), [0, 1])

    def test_modify_by_indices_out_of_range(self):
        """Test modify raises error on out of range index."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(10, 3, 5))
        
        with pytest.raises(IndexError):
            buffer.modify_by_indices(torch.randn(1, 3, 5), [100])

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(50, 3, 5))
        
        buffer.clear()
        
        assert len(buffer) == 0

    def test_getitem_int(self):
        """Test integer indexing."""
        buffer = Buffer(buffer_max_size=100)
        examples = torch.randn(10, 3, 5)
        buffer.add_examples(examples)
        
        result = buffer[0]
        
        assert result.shape == (3, 5)
        assert torch.allclose(result, examples[0])

    def test_getitem_slice(self):
        """Test slice indexing."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(10, 3, 5))
        
        result = buffer[2:5]
        
        assert result.shape == (3, 3, 5)

    def test_getitem_list(self):
        """Test list indexing."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(10, 3, 5))
        
        result = buffer[[0, 3, 7]]
        
        assert result.shape == (3, 3, 5)

    def test_getitem_tensor(self):
        """Test tensor indexing."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(10, 3, 5))
        
        result = buffer[torch.tensor([0, 3, 7])]
        
        assert result.shape == (3, 3, 5)

    def test_repr(self):
        """Test string representation."""
        buffer = Buffer(buffer_max_size=100)
        buffer.add_examples(torch.randn(50, 3, 5))
        
        repr_str = repr(buffer)
        
        assert "Buffer" in repr_str
        assert "50" in repr_str
        assert "100" in repr_str


class MockGwgModel(nn.Module):
    """Mock model for GWG sampler testing."""

    def __init__(self, k: int = 3, dx: int = 5):
        super().__init__()
        self.k = k
        self.l = k  # Alias used by DLP sampler
        self.dx = dx
        self.device = torch.device("cpu")
        self._linear = nn.Linear(k * dx, 1)  # Dummy parameter

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 2D indices to one-hot."""
        if x.dim() == 2:
            # (N, dx) -> (N, k, dx)
            return torch.nn.functional.one_hot(x.long(), self.k).float().permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        """Forward pass returning visible, intermediate, hidden."""
        visible = x
        intermediate = x.mean(dim=-1)  # (N, k)
        hidden = intermediate.mean(dim=-1, keepdim=True)  # (N, 1)
        return visible, intermediate, hidden

    def negative_energy(self, intermediate: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Compute negative energy."""
        return -intermediate.sum(dim=-1)  # (N,)


class TestGwgSampler:
    """Tests for the GwgSampler class."""

    def test_init(self):
        """Test GWG sampler initialization."""
        model = MockGwgModel(k=3, dx=5)
        sampler = GwgSampler(model, input_shape=(3, 5), max_len=100)
        
        assert sampler.model is model
        assert sampler.input_shape == (3, 5)
        assert sampler.max_len == 100
        assert sampler.oh_mode is False
        assert len(sampler.buffer) == 0

    def test_init_with_oh_mode(self):
        """Test initialization with oh_mode."""
        model = MockGwgModel()
        sampler = GwgSampler(model, input_shape=(3, 5), oh_mode=True)
        
        assert sampler.oh_mode is True

    def test_calculate_q_shape(self):
        """Test _calculate_q returns correct shape."""
        model = MockGwgModel(k=3, dx=5)
        sampler = GwgSampler(model, input_shape=(3, 5))
        
        # Create one-hot input with gradients
        inp = torch.randn(10, 3, 5, requires_grad=True)
        inp.retain_grad()
        
        # Mock forward pass to compute gradients
        loss = inp.sum()
        loss.backward()
        
        q = sampler._calculate_q(inp)
        
        # Should be (batch, l*dx) = (10, 15)
        assert q.shape == (10, 15)
        # Should be valid probability distribution
        assert torch.allclose(q.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_flip_dim_2dim_basic(self):
        """Test _flip_dim_2dim basic operation."""
        model = MockGwgModel(k=3, dx=5)
        sampler = GwgSampler(model, input_shape=(3, 5))
        
        inp = torch.zeros(4, 5)
        # Indices encode (l_value * dx + dx_position)
        # Index 7 = 1*5 + 2 -> set position 2 to value 1
        indices = torch.tensor([7, 7, 7, 7])
        
        result = sampler._flip_dim_2dim(inp, indices)
        
        # With 100% acceptance, position 2 should be 1.0
        assert result.shape == (4, 5)
        assert torch.all(result[:, 2] == 1.0)

    def test_flip_dim_2dim_with_acceptance(self):
        """Test _flip_dim_2dim with acceptance probabilities."""
        model = MockGwgModel(k=3, dx=5)
        sampler = GwgSampler(model, input_shape=(3, 5))
        
        torch.manual_seed(42)
        inp = torch.zeros(100, 5)
        indices = torch.full((100,), 7)
        accept_probs = torch.full((100,), 0.5)  # 50% acceptance
        
        result = sampler._flip_dim_2dim(inp, indices, accept_probs)
        
        # Some should be flipped, some not
        n_flipped = (result[:, 2] == 1.0).sum().item()
        assert 30 < n_flipped < 70  # Roughly 50% ± some variance

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        model = MockGwgModel()
        sampler = GwgSampler(model, input_shape=(3, 5))
        sampler.buffer.add_examples(torch.randn(50, 5))
        
        sampler.clear_buffer()
        
        assert len(sampler.buffer) == 0

    def test_repr(self):
        """Test string representation."""
        model = MockGwgModel()
        sampler = GwgSampler(model, input_shape=(3, 5), max_len=100, oh_mode=True)
        
        repr_str = repr(sampler)
        
        assert "GwgSampler" in repr_str
        assert "(3, 5)" in repr_str
        assert "100" in repr_str
        assert "True" in repr_str


class TestDlpSampler:
    """Tests for the DlpSampler class."""

    def test_init(self):
        """Test DLP sampler initialization."""
        model = MockGwgModel(k=3, dx=5)
        sampler = DlpSampler(model, input_shape=(3, 5), step_size=0.2, max_len=100)
        
        assert sampler.model is model
        assert sampler.input_shape == (3, 5)
        assert sampler.step_size == 0.2
        assert sampler.max_len == 100
        assert sampler.oh_mode is False
        assert len(sampler.buffer) == 0

    def test_init_eye_tensor(self):
        """Test that eye tensor is properly initialized."""
        model = MockGwgModel(k=4, dx=5)
        sampler = DlpSampler(model, input_shape=(4, 5))
        
        # Eye tensor should be (1, k, k, 1)
        assert sampler.eye_tensor.shape == (1, 4, 4, 1)
        # Should be identity matrix
        assert torch.allclose(
            sampler.eye_tensor.squeeze(), 
            torch.eye(4)
        )

    def test_calculate_q_regular_shape(self):
        """Test _calculate_q_regular returns correct shape."""
        model = MockGwgModel(k=3, dx=5)
        sampler = DlpSampler(model, input_shape=(3, 5))
        
        inp_detached = torch.randn(10, 3, 5)
        delta_f = torch.randn(10, 3, 5)
        
        q = sampler._calculate_q_regular(inp_detached, delta_f)
        
        # Should be (batch, k, dx) = (10, 3, 5)
        assert q.shape == (10, 3, 5)
        # Should be valid probability distribution over dim 1
        assert torch.allclose(q.sum(dim=1), torch.ones(10, 5), atol=1e-5)

    def test_calculate_q_safe_shape(self):
        """Test _calculate_q_safe returns correct shape."""
        model = MockGwgModel(k=3, dx=5)
        sampler = DlpSampler(model, input_shape=(3, 5))
        
        inp_detached = torch.randn(10, 3, 5)
        delta_f = torch.randn(10, 3, 5)
        
        q = sampler._calculate_q_safe(inp_detached, delta_f)
        
        # Should be (batch, k, dx) = (10, 3, 5)
        assert q.shape == (10, 3, 5)
        # Should be valid probability distribution
        assert torch.allclose(q.sum(dim=1), torch.ones(10, 5), atol=1e-5)

    def test_calculate_q_methods_equivalent(self):
        """Test that regular and safe methods give same results."""
        model = MockGwgModel(k=3, dx=5)
        sampler = DlpSampler(model, input_shape=(3, 5))
        
        torch.manual_seed(42)
        inp_detached = torch.randn(10, 3, 5)
        delta_f = torch.randn(10, 3, 5)
        
        q_regular = sampler._calculate_q_regular(inp_detached, delta_f)
        q_safe = sampler._calculate_q_safe(inp_detached, delta_f)
        
        assert torch.allclose(q_regular, q_safe, atol=1e-5)

    def test_clear_buffer(self):
        """Test clearing the buffer."""
        model = MockGwgModel()
        sampler = DlpSampler(model, input_shape=(3, 5))
        sampler.buffer.add_examples(torch.randn(50, 5))
        
        sampler.clear_buffer()
        
        assert len(sampler.buffer) == 0

    def test_repr(self):
        """Test string representation."""
        model = MockGwgModel()
        sampler = DlpSampler(
            model, input_shape=(3, 5), step_size=0.3, max_len=200, oh_mode=True
        )
        
        repr_str = repr(sampler)
        
        assert "DlpSampler" in repr_str
        assert "(3, 5)" in repr_str
        assert "0.3" in repr_str
        assert "200" in repr_str
        assert "True" in repr_str


class TestSamplersIntegration:
    """Integration tests for samplers."""

    def test_gwg_buffer_persistence(self):
        """Test that GWG sampler maintains buffer across calls."""
        model = MockGwgModel(k=3, dx=5)
        model.eval()  # Set to eval mode so buffer gets updated
        sampler = GwgSampler(model, input_shape=(3, 5), max_len=1000)
        
        # First call with empty buffer
        inputs = torch.randint(0, 3, (32, 5)).float()
        sampler.buffer.add_examples(inputs)  # Manually add to test
        
        # Buffer should have examples
        assert len(sampler.buffer) == 32
        
        # Add more
        sampler.buffer.add_examples(torch.randint(0, 3, (32, 5)).float())
        assert len(sampler.buffer) == 64

    def test_dlp_buffer_persistence(self):
        """Test that DLP sampler maintains buffer across calls."""
        model = MockGwgModel(k=3, dx=5)
        sampler = DlpSampler(model, input_shape=(3, 5), max_len=1000)
        
        # Add examples to buffer
        inputs = torch.randint(0, 3, (32, 5)).float()
        sampler.buffer.add_examples(inputs)
        
        assert len(sampler.buffer) == 32

    def test_samplers_import(self):
        """Test that all samplers can be imported from module."""
        from deem.core.samplers import Buffer, GwgSampler, DlpSampler
        
        assert Buffer is not None
        assert GwgSampler is not None
        assert DlpSampler is not None

    def test_samplers_import_from_core(self):
        """Test that samplers can be imported via core."""
        from deem.core import samplers
        
        assert hasattr(samplers, 'Buffer')
        assert hasattr(samplers, 'GwgSampler')
        assert hasattr(samplers, 'DlpSampler')

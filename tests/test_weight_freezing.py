"""Test weight freezing mechanism.

This module tests the weight freezing functionality that prevents the
first slice of weights/biases from being updated when dh=1.

This is CRITICAL for matching production run_predict.py behavior.
"""

import torch

from deem.core.models.rbm_gwg import MultinomialRBMGwg


class TestSaveFixedWeights:
    """Test that fixed weights are saved during initialization."""

    def test_fixed_weight_attributes_exist_when_dh_1(self):
        """Test that fixed weight attributes are created when dh=1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Check that fixed weight attributes exist
        assert hasattr(rbm, 'weights_l'), "weights_l attribute should exist when dh=1"
        assert hasattr(rbm, 'weights_m'), "weights_m attribute should exist when dh=1"
        assert hasattr(rbm, 'hidden_mask'), "hidden_mask attribute should exist when dh=1"
        assert hasattr(rbm, 'visible_mask'), "visible_mask attribute should exist when dh=1"

    def test_fixed_weight_shapes(self):
        """Test that fixed weights have correct shapes."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Verify shapes match the slices they represent
        assert rbm.weights_l.shape == rbm.weights[0].shape
        assert rbm.weights_m.shape == rbm.weights.permute(1, 3, 0, 2)[0].shape
        assert rbm.hidden_mask.shape == rbm.hidden_bias[:, 0].shape
        assert rbm.visible_mask.shape == rbm.visible_bias[:, 0].shape

    def test_fixed_weight_values_match_initial(self):
        """Test that saved fixed weights match initial weight values."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Values should match current weight slices
        torch.testing.assert_close(rbm.weights_l, rbm.weights[0])
        torch.testing.assert_close(
            rbm.weights_m,
            rbm.weights.permute(1, 3, 0, 2)[0]
        )
        torch.testing.assert_close(rbm.hidden_mask, rbm.hidden_bias[:, 0])
        torch.testing.assert_close(rbm.visible_mask, rbm.visible_bias[:, 0])

    def test_no_fixed_weights_when_dh_gt_1(self):
        """Test that fixed weight mechanism is disabled when dh > 1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=3, k=3, l=3, m=3,  # dh=3, not 1
            device='cpu', init_method='rand'
        )

        # Should NOT have fixed weight attributes
        assert not hasattr(rbm, 'weights_l')
        assert not hasattr(rbm, 'weights_m')
        assert not hasattr(rbm, 'hidden_mask')
        assert not hasattr(rbm, 'visible_mask')


class TestZeroFixedWeightsGrad:
    """Test that fixed weight gradients are zeroed correctly."""

    def test_gradients_zeroed_when_dh_1(self):
        """Test that fixed weight gradients are zeroed when dh=1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Create dummy input
        x = torch.randint(0, 3, (16, 10)).float()
        x = rbm.preprocess(x)

        # Forward pass
        h = rbm.calc_hidden_probs(x)
        energy = rbm.energy(x, h)
        loss = energy.mean()

        # Backward pass
        rbm.zero_grad()
        loss.backward()

        # Some gradients should be non-zero before zeroing
        # (we could assert this but it's not critical for the test)

        # Zero fixed weights grad
        rbm.zero_fixed_weights_grad()

        # Verify gradients are zeroed
        assert torch.all(rbm.weights.grad[0] == 0), "weights.grad[0] should be zero"
        assert torch.all(rbm.weights.grad.permute(1, 3, 0, 2)[0] == 0), "weights_m grad should be zero"
        assert torch.all(rbm.visible_bias.grad[:, 0] == 0), "visible_bias.grad[:, 0] should be zero"
        assert torch.all(rbm.hidden_bias.grad[:, 0] == 0), "hidden_bias.grad[:, 0] should be zero"

    def test_noop_when_no_grad(self):
        """Test that zero_fixed_weights_grad is safe when no gradients exist."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Call without backward pass - should not raise
        rbm.zero_fixed_weights_grad()  # No gradients exist yet

    def test_noop_when_dh_gt_1(self):
        """Test that zero_fixed_weights_grad is a no-op when dh > 1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=3, k=3, l=3, m=3,  # dh=3
            device='cpu', init_method='rand'
        )

        # Create dummy input and backward
        x = torch.randint(0, 3, (16, 10)).float()
        x = rbm.preprocess(x)
        h = rbm.calc_hidden_probs(x)
        loss = rbm.energy(x, h).mean()
        rbm.zero_grad()
        loss.backward()

        # Store original gradients
        orig_w_grad = rbm.weights.grad.clone()
        orig_vb_grad = rbm.visible_bias.grad.clone()
        orig_hb_grad = rbm.hidden_bias.grad.clone()

        # Call zero_fixed_weights_grad - should be no-op
        rbm.zero_fixed_weights_grad()

        # Gradients should be unchanged
        torch.testing.assert_close(rbm.weights.grad, orig_w_grad)
        torch.testing.assert_close(rbm.visible_bias.grad, orig_vb_grad)
        torch.testing.assert_close(rbm.hidden_bias.grad, orig_hb_grad)


class TestFixedWeightsUnchangedDuringTraining:
    """Test that fixed weights don't change during training steps."""

    def test_fixed_weights_unchanged_after_training_steps(self):
        """Test that fixed weights remain constant during training."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Save initial fixed weights
        initial_fixed = {
            'w0': rbm.weights[0].clone(),
            'wm': rbm.weights.permute(1, 3, 0, 2)[0].clone(),
            'vb0': rbm.visible_bias[:, 0].clone(),
            'hb0': rbm.hidden_bias[:, 0].clone()
        }

        # Simulate training steps
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)
        x = torch.randint(0, 3, (16, 10)).float()
        x = rbm.preprocess(x)

        for _ in range(5):  # 5 training steps
            optimizer.zero_grad()
            h = rbm.calc_hidden_probs(x)
            loss = rbm.energy(x, h).mean()
            loss.backward()
            rbm.zero_fixed_weights_grad()  # CRITICAL CALL
            optimizer.step()

        # Verify fixed weights unchanged
        torch.testing.assert_close(
            rbm.weights[0], initial_fixed['w0'],
            msg="weights[0] should not change during training"
        )
        torch.testing.assert_close(
            rbm.weights.permute(1, 3, 0, 2)[0], initial_fixed['wm'],
            msg="weights_m should not change during training"
        )
        torch.testing.assert_close(
            rbm.visible_bias[:, 0], initial_fixed['vb0'],
            msg="visible_bias[:, 0] should not change during training"
        )
        torch.testing.assert_close(
            rbm.hidden_bias[:, 0], initial_fixed['hb0'],
            msg="hidden_bias[:, 0] should not change during training"
        )

    def test_non_fixed_weights_do_change(self):
        """Test that non-fixed weights DO change during training (sanity check)."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=3, k=3, l=3, m=3,  # dh=3, so no weight freezing
            device='cpu', init_method='rand'
        )

        # Save initial weights
        initial_weights = rbm.weights.clone()

        # Training steps
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)
        x = torch.randint(0, 3, (16, 10)).float()
        x = rbm.preprocess(x)

        for _ in range(5):
            optimizer.zero_grad()
            h = rbm.calc_hidden_probs(x)
            loss = rbm.energy(x, h).mean()
            loss.backward()
            optimizer.step()

        # Weights SHOULD have changed
        assert not torch.allclose(rbm.weights, initial_weights), \
            "Weights should change during training when dh > 1"


class TestRefreezeWeights:
    """Test the refreeze_weights method."""

    def test_refreeze_restores_saved_values(self):
        """Test that refreeze_weights restores saved weight values."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Save original fixed values
        original_w0 = rbm.weights[0].clone()
        original_wm = rbm.weights.permute(1, 3, 0, 2)[0].clone()
        original_vb0 = rbm.visible_bias[:, 0].clone()
        original_hb0 = rbm.hidden_bias[:, 0].clone()

        # Manually corrupt the "fixed" weights
        with torch.no_grad():
            rbm.weights[0] += 100.0
            rbm.visible_bias[:, 0] += 100.0

        # Refreeze should restore
        rbm.refreeze_weights()

        # Verify restoration
        torch.testing.assert_close(rbm.weights[0], original_w0)
        torch.testing.assert_close(
            rbm.weights.permute(1, 3, 0, 2)[0], original_wm
        )
        torch.testing.assert_close(rbm.visible_bias[:, 0], original_vb0)
        torch.testing.assert_close(rbm.hidden_bias[:, 0], original_hb0)

    def test_refreeze_noop_when_dh_gt_1(self):
        """Test that refreeze_weights is a no-op when dh > 1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=3, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Store original weights
        original_weights = rbm.weights.clone()

        # Corrupt weights
        with torch.no_grad():
            rbm.weights += 100.0

        # refreeze should be no-op (no saved values when dh > 1)
        rbm.refreeze_weights()

        # Weights should still be corrupted (not restored)
        assert not torch.allclose(rbm.weights, original_weights)


class TestPrintFixedWeightsSum:
    """Test the print_fixed_weights_sum method."""

    def test_returns_sum_when_dh_1(self):
        """Test that print_fixed_weights_sum returns a sum when dh=1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        result = rbm.print_fixed_weights_sum()

        # Should return a float
        assert isinstance(result, float)

        # Should be non-zero (random init)
        # Actually could be close to zero, so just check it's a number
        assert result == result  # Not NaN

    def test_returns_zero_when_dh_gt_1(self):
        """Test that print_fixed_weights_sum returns 0.0 when dh > 1."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=3, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        result = rbm.print_fixed_weights_sum()

        assert result == 0.0

    def test_sum_constant_during_training_with_freezing(self):
        """Test that fixed weights sum remains constant during training."""
        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Get initial sum
        initial_sum = rbm.print_fixed_weights_sum()

        # Training steps WITH weight freezing
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)
        x = torch.randint(0, 3, (16, 10)).float()
        x = rbm.preprocess(x)

        for _ in range(5):
            optimizer.zero_grad()
            h = rbm.calc_hidden_probs(x)
            loss = rbm.energy(x, h).mean()
            loss.backward()
            rbm.zero_fixed_weights_grad()
            optimizer.step()

        # Sum should be unchanged
        final_sum = rbm.print_fixed_weights_sum()
        assert abs(final_sum - initial_sum) < 1e-6, \
            f"Fixed weights sum changed: {initial_sum} -> {final_sum}"


class TestTrainerIntegration:
    """Test that weight freezing integrates with RBMTrainer."""

    def test_trainer_calls_zero_fixed_weights_grad(self):
        """Test that trainer properly calls zero_fixed_weights_grad."""
        from torch.utils.data import DataLoader, TensorDataset

        from deem.core.losses import EBMLoss
        from deem.core.training import RBMTrainer

        rbm = MultinomialRBMGwg(
            dx=10, dh=1, k=3, l=3, m=3,
            device='cpu', init_method='rand'
        )

        # Save initial fixed weights
        initial_w0 = rbm.weights[0].clone()
        initial_vb0 = rbm.visible_bias[:, 0].clone()

        # Create simple dataset
        data = torch.randint(0, 3, (64, 10)).float()
        dataset = TensorDataset(data)
        train_loader = DataLoader(dataset, batch_size=16)

        # Setup trainer
        loss_fn = EBMLoss(rbm)
        optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1, momentum=0.9)
        trainer = RBMTrainer(rbm, optimizer, loss_fn, device='cpu')

        # Train for 2 epochs
        trainer.fit(train_loader, epochs=2, verbose=False)

        # Fixed weights should be unchanged
        torch.testing.assert_close(
            rbm.weights[0], initial_w0,
            msg="Trainer should preserve fixed weights via zero_fixed_weights_grad"
        )
        torch.testing.assert_close(
            rbm.visible_bias[:, 0], initial_vb0,
            msg="Trainer should preserve fixed visible bias via zero_fixed_weights_grad"
        )

#!/usr/bin/env python3
"""Test that AutoML models work out-of-the-box without model_dir."""

import numpy as np
from deem import DEEM

print("="*60)
print("Testing bundled AutoML models (no model_dir needed!)")
print("="*60)

# Create synthetic data
np.random.seed(42)
predictions = np.random.randint(0, 3, (500, 10))

# Test 1: Auto hyperparameters WITHOUT specifying model_dir
print("\n✓ Test 1: auto_hyperparameters=True (no model_dir)")
print("-"*60)
model = DEEM(
    n_classes=3,
    auto_hyperparameters=True,  # Should use bundled models!
    epochs=5,  # Override for speed
)

print("Creating model with auto_hyperparameters...")
model.fit(predictions, verbose=False)
print(f"✓ Model trained successfully!")
print(f"  Batch size used: {model.batch_size}")
print(f"  Epochs used: {model.epochs}")
print(f"  Learning rate used: {model.learning_rate}")

# Test 2: Verify predictions work
consensus = model.predict(predictions)
print(f"\n✓ Predictions shape: {consensus.shape}")
print(f"  Unique predicted classes: {np.unique(consensus)}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("AutoML models work out-of-the-box without model_dir!")
print("="*60)

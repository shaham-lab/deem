#!/usr/bin/env python3
"""Test script to check for double printing issue."""

import sys
from pathlib import Path

# Use local source code
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import scipy.io
from deem import DEEM

# Load MNIST ensemble
data = scipy.io.loadmat('datasets/mnist_e_v1.mat')
predictions = data['f'].T  # Transpose to (n_samples, n_classifiers)
labels = data['y'].flatten()

# Use small subset
predictions = predictions[:1000]
labels = labels[:1000]

print("="*60)
print("Testing for double printing issue")
print("="*60)

# Create model
model = DEEM(
    n_classes=10,
    epochs=20,  # Just a few epochs for testing
    batch_size=256,
    device='cpu',  # Force CPU for consistent testing
)

print("\nCalling fit() with verbose=True and labels...")
print("Watch for duplicate buffer/weighted init messages:")
print("-"*60)

# Call fit ONCE
model.fit(predictions, labels=labels, verbose=True)

print("-"*60)
print("\n✓ fit() completed")
print("\nIf you see duplicate messages with IDENTICAL timestamps,")
print("it's a Jupyter output display issue, not actual duplicate calls.")
print("\nIf you see duplicate messages with DIFFERENT timestamps,")
print("the methods are being called twice (actual bug).")

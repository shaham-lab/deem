import argparse
import json
import os
import pytest
import torch

from models.multinomial_rbm_gwg import MultinomialRBMGwg as GwgRBM

@pytest.fixture()
def args_data():
    config_file = os.path.join(os.path.dirname(__file__), "test_config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    
    args = argparse.Namespace(**dict(config))
    rbm = GwgRBM(**vars(args))

    return args, rbm

def test_flip_dim_2d(args_data):
    # Arrange
    args, gwg_rbm = args_data

    N, I, L = 10, 5, 3
    tensor = torch.ones(N, I) * 0.5
    indices = torch.tensor([2,6,10,3,8,5,1,13,14,9])

    tensor_flipped_dim = torch.tensor(
        [[0.5000, 0.5000, 0., 0.5000, 0.5000],
        [0.5000, 1., 0.5000, 0.5000, 0.5000],
        [2, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 0., 0.5000],
        [0.5000, 0.5000, 0.5000, 1, 0.5000],
        [1, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0., 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 2, 0.5000],
        [0.5000, 0.5000, 0.5000, 0.5000, 2],
        [0.5000, 0.5000, 0.5000, 0.5000, 1]])
    
    tensor_result = gwg_rbm.sampler.flip_dim_2dim(inp_inputs=tensor,
                                                  sampled_indices=indices)
    
    assert torch.all(tensor_result == tensor_flipped_dim)




def test_stam():
    assert 4 > 3
"""
import torch

# Assuming you have the following tensors
N, I, L = 10, 5, 3
batch_tensor = torch.ones(N, I) * 0.5
indices = torch.tensor([2,6,10,3,8,5,1,13,14,9])

# Calculate the indices
n_indices = torch.arange(N)
i_indices = indices // I
l_values = indices % I

# Update the batch_tensor
batch_tensor[n_indices, l_values] = i_indices.float()

# Check the result
print(batch_tensor)

tensor([2,6,10,3,8,5,1,13,14,9])
tensor([[0.5000, 0.5000, 0., 0.5000, 0.5000],
        [0.5000, 1., 0.5000, 0.5000, 0.5000],
        [2, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 0., 0.5000],
        [0.5000, 0.5000, 0.5000, 1, 0.5000],
        [1, 0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0., 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 2, 0.5000],
        [0.5000, 0.5000, 0.5000, 0.5000, 2],
        [0.5000, 0.5000, 0.5000, 0.5000, 1]])

0009000=================

# Assuming you have the following tensor
N = 10  # replace with your value of N
accept_prob = torch.rand(N)  # replace with your accept_prob vector

# Generate random numbers
random_numbers = torch.rand(N)

# Create a mask where the value is True if the random number is less than or equal to the corresponding accept_prob
mask = random_numbers <= accept_prob

"""
import argparse
import json
import os
import pytest
import torch

from custom_modules.multinomial import Multinomial
from models.multinomial_rbm_langevin_new import MultinomialRBMLangevin as LangevinRBM

@pytest.fixture()
def args_data():
    config_file = os.path.join(os.path.dirname(__file__), "test_config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    
    args = argparse.Namespace(**dict(config))
    rbm = LangevinRBM(**vars(args))

    return args, rbm


def test_lower_energy_for_sampled(args_data):
    # Arrange
    args, langevin_rbm = args_data
    random_tensor = torch.rand(args.n, args.k, args.dx)
    random_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True) 
    sampler = langevin_rbm.sampler
    energy_pre_sampling = langevin_rbm.print_energy_metrics(random_tensor)
    print(random_tensor)
    
    # Act
    sampled_inputs = sampler.generate_samples(sampler.model, random_tensor,learning_rate=0.05)
    energy_post_sampling = langevin_rbm.print_energy_metrics(sampled_inputs)
    print(sampled_inputs)

    # Assert
    assert energy_post_sampling < energy_pre_sampling

def test_multinomial_net_weights(args_data):
    # Arrange
    args, langevin_rbm = args_data
    m = Multinomial(args.k, args.k, args.dx, args.dx, one_hot = False, use_softmax = False,init_method='identity')
    m_real = Multinomial(args.k, args.k, args.dx, args.dx, one_hot = False, use_softmax = False,init_method='identity')
    
    langevin_rbm = LangevinRBM(**vars(args),multinomial_net = m)
    random_tensor = torch.rand(args.n, args.k, args.dx)
    random_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True) 
    sampler = langevin_rbm.sampler
    energy_pre_sampling = langevin_rbm.print_energy_metrics(random_tensor)
    
    # Act
    _ = sampler.generate_samples(sampler.model, random_tensor,learning_rate=0.05)
    

    # Assert
    assert torch.all(m_real.weights == m.weights)

def test_stam():
    assert 4 > 3
import argparse
import json
import pytest
import torch

from models.multinomial_rbm_manual_new import MultinomialRBMManual as MRBM
from models.multinomial_rbm_grad_new import MultinomialRBMAutograd as MRBMgrad


class TestMultinomialRBM:

    @pytest.fixture(scope='module')
    def args_data():
        config_file = "test_config.json"
        
        with open(config_file, "r") as f:
            config = json.load(f)
    
        args = argparse.Namespace(**dict(config))

        return args
    
    @pytest.fixture()
    def rbm_data(args_data):
        args = args_data

        # Set a fixed random state to generate reproducible test data
        torch.manual_seed(args.seed)

        pos_vis = torch.randn(args.n, args.l, args.dx)
        neg_vis = torch.randn(args.n, args.l, args.dx)
        pos_hid = torch.randn(args.n, args.m, args.dh)
        neg_hid = torch.randn(args.n, args.m, args.dh)

        return pos_vis, neg_vis, pos_hid, neg_hid 

    @pytest.fixture()
    def manual_rbm(args_data):
            args = args_data

            rbm = MRBM(**vars(args),)

            return rbm

    @pytest.fixture()
    def autograd_rbm(args_data):
            args = args_data

            rbm = MRBMgrad(**vars(args),)

            return rbm

    def test_preprocess():
         pass
    
    def test_einsum_get_visible():
         pass
    
    def test_einsum_get_hidden():
         pass
    
    def test_manual_weight_update():
         pass
    
    def test_fixed_weights_after_update():
         pass
    
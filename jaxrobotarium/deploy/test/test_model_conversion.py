import numpy as np
import torch
import unittest
from safetensors.flax import load_file

from jaxrobotarium.deploy.test.actor import RNNActor
from jaxrobotarium.deploy.test.jax_actor import RNNQNetwork
from jaxmarl.wrappers.baselines import load_params
from jaxrobotarium.deploy.deploy import flax_to_torch

class TestModelConversion(unittest.TestCase):
    def setUp(self):
        self.model_weights = "jaxmarl/environments/marbler/deploy/test/qmix_rnn_predator_capture_prey_seed0_vmap0.safetensors" # replace with path to model weights
        self.input_dim = 38  # replace with input dimension
        self.hidden_dim = 512 # replace with hidden dimension
        self.output_dim = 5 # replace with output dimension

        # load jax actor
        self.params = load_params(self.model_weights)['agent']
        self.jax_actor = RNNQNetwork(
            action_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
        )

        # load torch actor
        self.flax_weights = load_file(self.model_weights)
        self.torch_actor = RNNActor(self.input_dim, self.output_dim, self.hidden_dim)
        self.torch_actor.load_state_dict(
            flax_to_torch(self.flax_weights, self.torch_actor.state_dict())
        )
    
    def test_forward(self):
        """Test forward pass of both actors align"""

        np.random.seed(0)

        hstate = np.zeros((1, 1, self.hidden_dim)).astype(np.float32)
        obs = np.random.rand(1, 1, self.input_dim).astype(np.float32)
        dones = np.zeros((1,1))

        jax_hidden, jax_q = self.jax_actor.apply(self.params, hstate, obs, dones)
        torch_q, torch_hidden = self.torch_actor(
            torch.from_numpy(obs[0]).to(torch.float32),
            torch.from_numpy(hstate[0]).to(torch.float32)
        )

        # print(jax_q.squeeze())
        # print(torch_q.detach().numpy().squeeze())

        self.assertTrue(
            np.allclose(jax_hidden.squeeze(), torch_hidden.detach().numpy().squeeze(), atol=1e-4)
        )
        self.assertTrue(
            np.allclose(jax_q.squeeze(), torch_q.detach().numpy().squeeze(), atol=1e-4)
        )
        

        
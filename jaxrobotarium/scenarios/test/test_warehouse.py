import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.warehouse import Warehouse

VISUALIZE = False

class TestWarehouse(unittest.TestCase):
    """unit tests for test_warehouse.py"""

    def setUp(self):
        self.num_agents = 2
        self.batch_size = 10
        self.env = Warehouse(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'class',
                'obs_type': 'class',
                'values': [[1, 0], [0, 1]],
                'sample': False
            },
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(~jnp.all(state.payload))
        self.assertTrue(state.zone1_load == 0)
        self.assertTrue(state.zone2_load == 0)
        self.assertTrue(state.p_pos.shape == (self.num_agents, 3))
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1.25, -0.5, 0], [-1.25, -0.5, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        # check number delivered updates for red zone and payloads update
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, 0])))
        self.assertAlmostEqual(new_state.zone1_load - state.zone1_load, 0)
        self.assertAlmostEqual(new_state.zone2_load - state.zone2_load, 1)

        state = new_state
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
        
    def test_reward(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1.25, -0.5, 0], [-1.25, -0.5, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], self.env.load_shaping+self.env.dropoff_shaping)
        self.assertEqual(rewards['agent_1'], self.env.load_shaping+self.env.dropoff_shaping)

        state = new_state
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0.0)
        self.assertEqual(rewards['agent_1'], 0.0)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1.25, 0.5, 0], [-1.25, 0.5, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        obs = self.env.get_obs(state)
        
        # agent 0
        expected_obs = jnp.array([1.25, 0.5, 0, -1.25, 0.5, 0])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'][:-self.env.het_manager.dim_h], expected_obs)
        )

        # agent 1
        expected_obs = jnp.array([-1.25, 0.5, 0, 1.25, 0.5, 0])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'][:-self.env.het_manager.dim_h], expected_obs)
        )
    
    def test_initialize_robotarium_state(self):
        state = self.env.initialize_robotarium_state(self.key)
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(~jnp.all(state.payload))
        self.assertTrue(state.p_pos.shape == (self.num_agents, 3))
        self.assertTrue(state.step == 0)
    
    def test_batched_rollout(self):
        self.env = Warehouse(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'class',
                'obs_type': 'class',
                'values': [[1, 0], [0, 1]],
                'sample': False
            },
            controller = {
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            }
        )
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state
        payload = jnp.full((self.batch_size, self.num_agents), 1)
        state = state.replace(
            payload = payload
        )

        def get_action(state):
            return {str(f'agent_{i}'): jnp.array([3]) for i in range(self.num_agents)}
        
        def wrapped_step(poses, unused):
            actions = jax.vmap(get_action, in_axes=(0))(poses)
            new_obs, new_state, rewards, dones, infos = jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, poses, actions)
            return new_state, (new_state, rewards)

        final_state, (batch, rewards) = jax.lax.scan(wrapped_step, state, None, 70)

        rewards = jnp.array([rewards[agent] for agent in rewards])
        
        # check that the robot moved
        for i in range(self.num_agents):
            self.assertGreater(
                jnp.sqrt(jnp.sum((final_state.p_pos[i] - initial_state.p_pos[0])**2)),
                0
            )
        
        if VISUALIZE:
            # hack to add extra dim
            render_batch = State()
            fields = {}
            for attr in batch.__dict__.keys():
                if getattr(batch, attr) is None:
                    continue
                fields[f'{attr}'] = getattr(batch, attr)[None, ...]
            render_batch = render_batch.replace(**fields)
            frames = self.env.render(render_batch, seed_index=0, env_index=0)
            frames[0].save(
                'jaxmarl/environments/marbler/scenarios/test/warehouse.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
        

if __name__ == '__main__':
    unittest.main()
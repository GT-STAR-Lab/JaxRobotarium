import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.arctic_transport import ArcticTransport

VISUALIZE = False

class TestArcticTransport(unittest.TestCase):
    """unit tests for arctic_transport.py"""

    def setUp(self):
        self.num_agents = 4
        self.batch_size = 10
        self.env = ArcticTransport(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'class',
                'obs_type': 'class',
                'values': [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'sample': False
            },
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))

        self.assertTrue(state.grid.shape == (8, 12))
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(state.p_pos.shape == (self.num_agents, 3))
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        _, state = self.env.reset(self.key)
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])

        state = new_state
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.8, 0], [-0.5, 0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        for i in range(self.num_agents):
            self.assertTrue(infos['success_rate'][i] == 1)

    def test_info(self):
        _, state = self.env.reset(self.key)
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.8, 0], [-0.5, 0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.all(infos['success_rate']))

        _, state = self.env.reset(self.key)
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.74, 0], [-0.5, 0.74, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertFalse(jnp.all(infos['success_rate']))

        _, state = self.env.reset(self.key)
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.5, 0], [-1.25, 0.5, 0], [0.5, 0.8, 0], [-0.5, 0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.all(infos['success_rate']))
    
    def test_decode_discrete_action(self):
        _, state = self.env.reset(self.key)
        grid = jnp.concatenate([jnp.ones((8, 6))*2, jnp.ones((8, 6))], axis=-1)

        # place agents on suboptimal cells
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [-0.5, 0, 0], [0.5, 0, 0]]),
            grid = grid,
        )
        actions = {str(f'agent_{i}'): self.env._decode_discrete_action(i, 1, state) for i in range(self.num_agents)}

        # check that action for agents 2 and 3 is suboptimal
        self.assertTrue(jnp.array_equal(actions['agent_2'][1], 0.1))
        self.assertTrue(jnp.array_equal(actions['agent_3'][1], 0.1))
        
        # place agents on optimal cells
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0, 0], [-0.5, 0, 0]]),
            grid = grid,
        )
        actions = {str(f'agent_{i}'): self.env._decode_discrete_action(i, 1, state) for i in range(self.num_agents)}

        # check that action for agents 2 and 3 is optimal
        self.assertTrue(jnp.array_equal(actions['agent_2'][1], 0.3))
        self.assertTrue(jnp.array_equal(actions['agent_3'][1], 0.3))

    def test_reward(self):
        _, state = self.env.reset(self.key)
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, -0.8, 0], [-0.5, -0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(rewards['agent_0'] < self.env.dist_shaping)

        state = new_state
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.8, 0], [-0.5, 0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0.0)
        self.assertEqual(rewards['agent_1'], 0.0)

        self.env.time_shaping = -1
        self.env.dist_shaping = 0
        state = new_state
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.8, 0], [-0.5, 0.8, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0.0)
        self.assertEqual(rewards['agent_1'], 0.0)

        state = new_state
        state = state.replace(
            p_pos = jnp.array([[1.25, 0.8, 0], [-1.25, 0.8, 0], [0.5, 0.74, 0], [-0.5, 0.74, 0]]),
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], -1)
        self.assertEqual(rewards['agent_1'], -1)

    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        state = state.replace(
            grid = jnp.arange(8*12).reshape(8, 12), # unique grid values for testing
        )
        obs = self.env.get_obs(state)

        drone1_obs = jnp.array([75, 76, 77, 87, 88, 89, -1, -1, -1])
        drone2_obs = jnp.array([78, 79, 80, 90, 91, 92, -1, -1, -1])
        
        # check drone obs are loaded correctly
        self.assertTrue(
            jnp.array_equal(obs['agent_0'][-(self.env.het_manager.dim_h+18):-(self.env.het_manager.dim_h+9)], drone1_obs)
        )
        self.assertTrue(
            jnp.array_equal(obs['agent_0'][-(self.env.het_manager.dim_h+9):-(self.env.het_manager.dim_h)], drone2_obs)
        )

    def test_batched_rollout(self):
        self.env = ArcticTransport(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'class',
                'obs_type': 'class',
                'values': [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
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
                jnp.sqrt(jnp.sum((final_state.p_pos[i] - initial_state.p_pos[i])**2)),
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
                'jaxmarl/environments/marbler/scenarios/test/arctic_transport.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
        

if __name__ == '__main__':
    unittest.main()
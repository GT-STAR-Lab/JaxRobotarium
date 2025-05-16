import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.material_transport import MaterialTransport

VISUALIZE = False

class TestMaterialTransport(unittest.TestCase):
    """unit tests for test_material_transport.py"""

    def setUp(self):
        self.num_agents = 2
        self.batch_size = 10
        self.env = MaterialTransport(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'capability_set',
                'obs_type': 'full_capability_set',
                'values': [[.45, 5], [.15, 15]],
                'sample': False
            },
            zone1_dist = {
                'mu': 50,
                'sigma': 1
            },
            zone2_dist = {
                'mu': 10,
                'sigma': 1
            }
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(~jnp.all(state.payload))
        self.assertTrue(state.zone1_load > 0)
        self.assertTrue(state.zone2_load > 0)
        self.assertTrue(state.p_pos.shape == (self.num_agents, 3))
        self.assertTrue(state.step == 0)
    
    def test_decode_action(self):
        _, state = self.env.reset(self.key)
        actions = {str(f'agent_{i}'): self.env._decode_discrete_action(i, jnp.array([1]), state) for i in range(self.num_agents)}

        # agent 0
        expected_action = state.p_pos[0, :2] + state.het_rep[0, 0] * jnp.array([[0, 1]])
        self.assertTrue(jnp.array_equal(expected_action, actions['agent_0']))

        # agent 1
        expected_action = state.p_pos[1, :2] + state.het_rep[1, 0] * jnp.array([[0, 1]])
        self.assertTrue(jnp.array_equal(expected_action, actions['agent_1'])) 
    
    def test_step(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.0, 0, 0], [-1.25, 0, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        # check payloads and loads update for zone 1
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, 0])))
        self.assertAlmostEqual(state.zone1_load - new_state.zone1_load, state.het_rep[0, 1])
        self.assertAlmostEqual(state.zone2_load - new_state.zone2_load, 0)

        state = new_state
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        # check payloads and loads update for zone 2
        p_pos = jnp.array([[1.25, 0, 0], [-1.25, 0, 0]])
        state = new_state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, 0])))
        self.assertAlmostEqual(state.zone1_load - new_state.zone1_load, 0)
        self.assertAlmostEqual(state.zone2_load - new_state.zone2_load, state.het_rep[0, 1])

        # check payload remains the same and loads remain the same
        state = new_state
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, 0])))
        self.assertAlmostEqual(state.zone1_load, new_state.zone1_load)
        self.assertTrue(state.zone2_load - new_state.zone2_load == 0)

        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
        
    def test_reward(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.0, 0, 0], [-1.25, 0, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 1)
        self.assertEqual(rewards['agent_1'], 1)

        state = new_state
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0.0)
        self.assertEqual(rewards['agent_1'], 0.0)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.0, 0, 0], [-1.25, 0, 0]])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([0, 1])
        )
        obs = self.env.get_obs(state)
        
        # agent 0
        expected_obs = jnp.array([0.0, 0, 0, -1.25, 0, 0, state.zone1_load[0], state.zone2_load[0]])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'][:-self.env.het_manager.dim_h], expected_obs)
        )

        # agent 0
        expected_obs = jnp.array([-1.25, 0, 0, 0.0, 0, 0, state.zone1_load[0], state.zone2_load[0]])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'][:-self.env.het_manager.dim_h], expected_obs)
        )
    
    def test_batched_rollout(self):
        self.env = MaterialTransport(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'capability_set',
                'obs_type': 'full_capability_set',
                'values': [[.45, 5], [.15, 15]],
                'sample': False
            },
            zone1_dist = {
                'mu': 50,
                'sigma': 1
            },
            zone2_dist = {
                'mu': 10,
                'sigma': 1
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
                jnp.sqrt(jnp.sum((final_state.p_pos.T[i][0] - initial_state.p_pos.T[i][0])**2)),
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
                'jaxmarl/environments/marbler/scenarios/test/mt.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
        

if __name__ == '__main__':
    unittest.main()
import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.foraging import Foraging

VISUALIZE = False

class TestForaging(unittest.TestCase):
    """unit tests for foraging.py"""

    def setUp(self):
        self.num_agents = 3
        self.num_resources = 2
        self.batch_size = 10
        self.env = Foraging(
            num_agents=self.num_agents,
            num_resources=self.num_resources,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'capability_set',
                'obs_type': 'full_capability_set',
                'values': [[1], [2], [3]],
                'sample': False
            }
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(jnp.all(state.payload <= jnp.sum(state.het_rep)))
        self.assertTrue(state.p_pos.shape == (self.num_agents + self.num_resources, 3))
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        _, state = self.env.reset(self.key)

        # check that payload updates correctly when correct leveled agents are in range
        p_pos = jnp.zeros(state.p_pos.shape)
        p_pos = p_pos.at[-1].set([1, 1, 1])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([6, 4])
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 4])))

        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])

        # check payload does not update if agents are underleveled
        state = new_state
        p_pos = jnp.zeros(state.p_pos.shape)
        p_pos = p_pos.at[-1].set([1, 1, 1])
        p_pos = p_pos.at[2].set([1, 1, 1])
        state = state.replace(
            p_pos = p_pos,
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 4])))

        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
    
    def test_reward(self):
        _, state = self.env.reset(self.key)

        # foraged
        p_pos = jnp.zeros(state.p_pos.shape)
        p_pos = p_pos.at[-1].set([1, 1, 1])
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([6, 4])
        )
        rewards = self.env.rewards(state)

        self.assertEqual(rewards['agent_0'], 6)
        self.assertEqual(rewards['agent_1'], 6)
        self.assertEqual(rewards['agent_2'], 6)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)

        # construct test observation
        p_pos = jnp.arange(self.num_agents+self.num_resources).reshape(-1,1).repeat(3, 1)
        state = state.replace(
            p_pos = p_pos,
            payload = jnp.array([6, 4])
        )
        obs = self.env.get_obs(state)
        self.assertEqual(len(obs), self.num_agents)
        
        # agent 0
        expected_obs = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 6, 4, 4, 4])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'][:-self.env.het_manager.dim_h], expected_obs)
        )

        # agent 1
        expected_obs = jnp.array([1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 6, 4, 4, 4])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'][:-self.env.het_manager.dim_h], expected_obs)
        )

        # agent 2
        expected_obs = jnp.array([2, 2, 2, 0, 0, 0, 1, 1, 1, 3, 3, 6, 4, 4, 4])
        self.assertTrue(
            jnp.array_equal(obs['agent_2'][:-self.env.het_manager.dim_h], expected_obs)
        )
    
    def test_batched_rollout(self):
        self.env = Foraging(
            num_agents=self.num_agents,
            num_resources=self.num_resources,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
            heterogeneity={
                'type': 'capability_set',
                'obs_type': 'full_capability_set',
                'values': [[1], [2], [3]],
                'sample': False
            },
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            }
        )
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state

        def get_action(state):
            return {str(f'agent_{i}'): jax.random.choice(self.key, jnp.arange(5)) for i in range(self.num_agents)}
        
        def wrapped_step(poses, unused):
            actions = jax.vmap(get_action, in_axes=(0))(poses)
            new_obs, new_state, rewards, dones, infos = jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, poses, actions)
            return new_state, (new_state, rewards)

        final_state, (batch, rewards) = jax.lax.scan(wrapped_step, state, None, 75)

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
                'jaxmarl/environments/marbler/scenarios/test/forage.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
        

if __name__ == '__main__':
    unittest.main()
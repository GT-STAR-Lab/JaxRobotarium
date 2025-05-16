import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.predator_prey import PredatorPrey

VISUALIZE = False

class TestPredatorPrey(unittest.TestCase):
    """unit tests for predator_prey.py"""

    def setUp(self):
        self.num_agents = 2
        self.num_landmarks = 2
        self.batch_size = 10
        self.env = PredatorPrey(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=80,
            update_frequency=1,
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(state.landmark_tagged == 0)
        self.assertTrue(state.p_pos.shape == (self.num_agents + 1, 3))
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        _, state = self.env.reset(self.key)
        self.env.prey_step = 0  # immobilize prey for test
        p_pos = jnp.array([[-0.5, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(new_state.landmark_tagged == 1)
        
        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
    
    def test_reward(self):
        _, state = self.env.reset(self.key)

        # one agent tags
        p_pos = jnp.array([[-0.5, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        rewards = self.env.rewards(state)

        self.assertEqual(rewards['agent_0'], 10)
        self.assertEqual(rewards['agent_1'], 10)

        # two agents tag
        p_pos = jnp.array([[-0.5, 0, 0], [-0.5, 0, 0], [-0.5, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        rewards = self.env.rewards(state)
        self.assertEqual(rewards['agent_0'], 10)
        self.assertEqual(rewards['agent_1'], 10)

        # no tag
        p_pos = jnp.array([[-0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        rewards = self.env.rewards(state)
        self.assertEqual(rewards['agent_0'], 0)
        self.assertEqual(rewards['agent_1'], 0)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)

        p_pos = jnp.array([[-0.5, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
        state = state.replace(
            p_pos = p_pos,
        )
        obs = self.env.get_obs(state)
        self.assertEqual(len(obs), self.num_agents)
        
        # agent 0
        expected_obs = jnp.array([-0.5, 0, 0, 0.5, 0, 0, -0.5, 0, 0])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'], expected_obs)
        )

        # agent 0
        expected_obs = jnp.array([0.5, 0, 0, -0.5, 0, 0, -0.5, 0, 0])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'], expected_obs)
        )
    
    def test_batched_rollout(self):
        self.env = PredatorPrey(
            num_agents=self.num_agents,
            action_type="Discrete",
            max_steps=80,
            update_frequency=1,
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
                'jaxmarl/environments/marbler/scenarios/test/predprey.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
        

if __name__ == '__main__':
    unittest.main()
import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.scenarios.navigation import Navigation

VISUALIZE = False

class TestNavigation(unittest.TestCase):
    """unit tests for navigation.py"""

    def setUp(self):
        self.num_agents = 3
        self.batch_size = 10
        self.env = Navigation(num_agents=self.num_agents, action_type="Continuous", max_steps=250, update_frequency=1)
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(state.p_pos.shape == (self.num_agents*2, 3))
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        _, state = self.env.reset(self.key)

        p_pos = jnp.array([[-1., 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([1, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        
        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
    
    def test_reward(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        rewards = self.env.rewards(state)
        self.assertEqual(rewards['agent_0'], 1 * self.env.pos_shaping)
        self.assertEqual(rewards['agent_1'], 1 * self.env.pos_shaping)
        self.assertEqual(rewards['agent_2'], 0)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        observations = self.env.get_obs(state)
        self.assertEqual(len(observations), self.num_agents)
        for i in range(self.num_agents):
            self.assertTrue(
                jnp.array_equal(
                    observations[str(f'agent_{i}')], 
                    jnp.concatenate((state.p_pos[i], state.p_pos[self.num_agents+i, :2]-state.p_pos[i, :2]))
                )
            )
    
    def test_initialize_robotarium_state(self):
        state = self.env.initialize_robotarium_state(self.key)
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(state.p_pos.shape == (self.num_agents*2, 3))
        self.assertTrue(state.step == 0)
    
    def test_batched_rollout(self):
        self.env = Navigation(
            num_agents=2,
            action_type="Discrete",
            max_steps=75,
            update_frequency=30,
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            }
        )
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state

        def get_action(state):
            goal_pos = state.p_pos[self.env.num_agents:, :2]
            actions = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
            dir_to_goal = goal_pos - state.p_pos[:self.env.num_agents, :2]
            dir_to_goal = dir_to_goal / jnp.linalg.norm(dir_to_goal, axis=1)[:, None]
            dots = jax.vmap(jnp.dot, in_axes=(None, 0))(actions, dir_to_goal)
            best_action = jnp.argmax(dots, axis=1)
            return {str(f'agent_{i}'): best_action[i] for i in range(self.env.num_agents)}
        
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
            self.env.render(batch.p_pos[:, 1, ...], name='navigation env 0', save_path='jaxmarl/environments/marbler/scenarios/test/navigation.gif')
        

if __name__ == '__main__':
    unittest.main()
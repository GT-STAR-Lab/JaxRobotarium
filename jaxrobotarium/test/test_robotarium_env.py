import unittest
import jax
import jax.numpy as jnp
from jaxrobotarium.robotarium_env import *

class MockEnv(RobotariumEnv):
        def __init__(self, num_agents, max_steps=50, **kwargs):
            self.het_manager = HetManager(num_agents, **kwargs.get('heterogeneity'))
            super().__init__(num_agents, max_steps, **kwargs)
        
        def reset(self, key):
            state = State(
                p_pos=jnp.zeros((self.num_agents, 3)),
                done=jnp.full((self.num_agents), False),
                step=0,
                het_rep = self.het_manager.sample(key)
            )

            return self.get_obs(state), state         
        
        def get_obs(self, state):
            obs = {a: self.het_manager.process_obs(state.p_pos[i], state, i) for i, a in enumerate(self.agents)}
            return obs    

class TestRobotariumEnv(unittest.TestCase):
    """unit tests for RobotariumEnv class"""

    def setUp(self):
        self.num_agents = 3
        self.batch_size = 10
        self.env = RobotariumEnv(num_agents=self.num_agents, action_type="Continuous")
        self.key = jax.random.PRNGKey(0)
    
    def test_get_violations(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        violations = self.env._get_violations(state)
        self.assertEqual(violations['collision'], 3)

    def test_observation_space(self):
        self.env.observation_spaces = {str(i): "obs_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.observation_space(str(i)), "obs_space")

    def test_action_space(self):
        self.env.action_spaces = {str(i): "act_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.action_space(str(i)), "act_space")
    
    def test_discrete_action_decoder(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        decoded_actions = [self.env._decode_discrete_action(i, i+1, state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] >= -1.6 and decoded_actions[i][0] <= 1.6)
            self.assertTrue(decoded_actions[i][1] >= -1 and decoded_actions[i][1] <= 1)
        
        state = State(
            p_pos = jnp.array([[0.0, 1.1, 0.0], [0.0, -1.1, 0.0], [1.7, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        decoded_actions = [self.env._decode_discrete_action(i, i+1, state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] >= -1.6 and decoded_actions[i][0] <= 1.6)
            self.assertTrue(decoded_actions[i][1] >= -1 and decoded_actions[i][1] <= 1)
    
    def test_continuous_action_decoder(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        actions = jnp.array([[0, 0], [0.1, 0.1], [0.2, 0.2]])
        decoded_actions = [self.env._decode_continuous_action(i, actions[i], state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] == actions[i][0])
            self.assertTrue(decoded_actions[i][1] == actions[i][1])
    
    def test_robotarium_step(self):
        poses = jnp.array([[0., 0, 0]])
        goals = jnp.array([[1., 0]])
        self.env = RobotariumEnv(
            num_agents=self.num_agents,
            action_type="Discrete",
            controller={"controller": "clf_uni_position"},
            update_frequency=30
        )
        final_pose = self.env._robotarium_step(poses, goals)
        self.assertTrue(jnp.linalg.norm(poses - final_pose) > 0.1)

class TestHetManager(unittest.TestCase):
    """unit tests for HetManager"""   

    def test_id(self):
        args = {
            'num_agents': 3,
            'heterogeneity': {
                'type': 'id',
                'obs_type': 'id',
                'values': None
            }
        }
        env = MockEnv(**args)
        obs, state = env.reset(jax.random.PRNGKey(0))

        for i, (agent, o) in enumerate(obs.items()):
            self.assertTrue(jnp.array_equal(o[-3:], jnp.eye(args['num_agents'])[i]))
        
        self.assertTrue(jnp.array_equal(state.het_rep, jnp.eye(args['num_agents'])))
    
    def test_class(self):
        args = {
            'num_agents': 3,
            'heterogeneity': {
                'type': 'class',
                'obs_type': 'class',
                'values': [[0, 0], [1, 0]],
                'sample': True
            }
        }
        env = MockEnv(**args)
        obs, state = env.reset(jax.random.PRNGKey(0))

        for i, (agent, o) in enumerate(obs.items()):
            self.assertTrue(
                jnp.array_equal(o[-2:], jnp.array([0, 0])) \
                or jnp.array_equal(o[-2:], jnp.array([1, 0]))                  
            )
        
        self.assertTrue(state.het_rep.shape[0] == args['num_agents'])
    
    def test_capability_set(self):
        args = {
            'num_agents': 3,
            'heterogeneity': {
                'type': 'capability_set',
                'obs_type': 'capability_set',
                'values': [[0, 1.], [1., 0]],
                'sample': True
            }
        }
        env = MockEnv(**args)
        obs, state = env.reset(jax.random.PRNGKey(0))

        for i, (agent, o) in enumerate(obs.items()):
            self.assertTrue(
                jnp.array_equal(o[-2:], env.het_manager.representation_set[0]) \
                or jnp.array_equal(o[-2:], env.het_manager.representation_set[1])
            )
        
        self.assertTrue(state.het_rep.shape[0] == args['num_agents'])
    
    def test_full_capability_set(self):
        args = {
            'num_agents': 3,
            'heterogeneity': {
                'type': 'capability_set',
                'obs_type': 'full_capability_set',
                'values': [[0, 1.], [1., 0]],
                'sample': True
            }
        }
        env = MockEnv(**args)
        obs, state = env.reset(jax.random.PRNGKey(0))

        obs = jnp.array([o for _, o in obs.items()])

        # agent 0
        expected_het = jnp.concatenate([state.het_rep[0], state.het_rep[1], state.het_rep[2]])
        self.assertTrue(jnp.array_equal(obs[0, -env.het_manager.dim_h:], expected_het))

        # agent 1
        expected_het = jnp.concatenate([state.het_rep[1], state.het_rep[2], state.het_rep[0]])
        self.assertTrue(jnp.array_equal(obs[1, -env.het_manager.dim_h:], expected_het))

        # agent 2
        expected_het = jnp.concatenate([state.het_rep[2], state.het_rep[0], state.het_rep[1]])
        self.assertTrue(jnp.array_equal(obs[2, -env.het_manager.dim_h:], expected_het))
        
        self.assertTrue(state.het_rep.shape[0] == args['num_agents'])
    
    def test_unaware(self):
        args = {
            'num_agents': 3,
            'heterogeneity': {
                'type': 'capability_set',
                'obs_type': None,
                'values': [[0, 1.], [1., 0]],
                'sample': True
            }
        }
        env = MockEnv(**args)
        obs, state = env.reset(jax.random.PRNGKey(0))

        for i, (agent, o) in enumerate(obs.items()):
            self.assertTrue(jnp.array_equal(o, jnp.zeros(args['num_agents'])))
        
        self.assertTrue(state.het_rep.shape[0] == args['num_agents'])
        

if __name__ == '__main__':
    unittest.main()
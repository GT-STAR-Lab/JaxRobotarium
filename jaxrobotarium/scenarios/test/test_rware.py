import unittest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.rware import RWARE

VISUALIZE = False

class TestRWARE(unittest.TestCase):
    """unit tests for test_rware.py"""

    def setUp(self):
        self.num_agents = 2
        self.num_cells = 4
        self.batch_size = 10
        self.env = RWARE(
            num_agents=self.num_agents,
            num_cells=self.num_cells,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(jnp.array_equal(state.payload, jnp.full((self.num_agents,), -1)))
        self.assertTrue(state.p_pos.shape == (self.num_agents+self.env.num_cells, 3))
        self.assertTrue(state.grid.shape == (self.env.num_cells, 3))
        self.assertTrue(state.request.shape == (self.num_agents,))
        self.assertTrue(jnp.unique(state.request).shape[0] == self.num_agents)
        self.assertTrue(state.step == 0)

    def test_decode_discrete_action(self):
        # action should execute fine and cart should be picked up
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, 0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        grid = jnp.array([[0.25, 0.25,  0], [.25, -0.25, 1], [0.75, 0.25,  2], [.75, -0.25, 3]])
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 1, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, -1])))

        # action should not be in collision with cart
        state = new_state
        p_pos = jnp.array(
            [[0.25, -0.28,  jnp.pi/2], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        actions = {str(f'agent_{i}'): jnp.array([1]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        self.assertTrue((jnp.abs(new_state.p_pos[0, :2] - p_pos[0, :2]) < jnp.array([0.125, 0.125])).all())

    
    def test_step(self):
        # agent 0 picks up shelf 0
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, 0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        grid = jnp.array([[0.25, 0.25,  0], [.25, -0.25, 1], [0.75, 0.25,  2], [.75, -0.25, 3]])
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 1, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, -1])))

        # agent 0 picks up shelf 1
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, -0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([0, -1, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, -1])))

        # both agents attempt to pick up shelf 0
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, 0.25,  0], [0.25, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 1, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, -1])))

        # both agents attempt to pick up unique shelves
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, 0.25,  0], [0.25, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, -1, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, 1])))

        # agent 0 drops off shelf, make sure agent 2 doesn't drop off shelf
        state = new_state
        p_pos = jnp.array(
            [[1.5, 0.25,  0], [-1, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
            request=jnp.array([0, 1]),
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(state.payload, jnp.array([0, 1])))
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, -1, 2, 3])))
        self.assertTrue(new_state.request[-1] == 1) # second shelf should stay the same
        self.assertTrue(new_state.request[0] != state.request[0]) # first shelf should change

        # agent 0 returns shelf 0 at cell 1
        prev_state = new_state
        state = new_state
        p_pos = jnp.array(
            [[0.25, -0.25,  0], [-1, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 0, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))

        # agent 0 attempts to return shelf at occupied cell
        state = prev_state
        p_pos = jnp.array(
            [[0.75, -0.25,  0], [-1, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, 1])))
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, -1, 2, 3])))

        # both agents attempt to return at same cell
        state = prev_state
        p_pos = jnp.array(
            [[0.25, -0.25,  0], [0.25, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 0, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))

        # both agents attempt to successfully return
        state = prev_state
        p_pos = jnp.array(
            [[0.25, -0.25,  0], [0.25, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, -1])))
        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([1, 0, 2, 3])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, -1])))

    def test_reward(self):
        # NOTE: this test is hacky because we unify step_env and reward in rware
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        _, state = self.env.reset(self.key)
        p_pos = jnp.array(
            [[0.25, 0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        grid = jnp.array([[0.25, 0.25,  0], [.25, -0.25, 1], [0.75, 0.25,  2], [.75, -0.25, 3]])
        state = state.replace(
            p_pos=p_pos,
            grid=grid
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0)
        self.assertEqual(rewards['agent_1'], 0)

        # agent 0 drops off shelf
        state = new_state
        p_pos = jnp.array(
            [[1.5, 0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0], [0.75, 0.25,  0], [.75, -0.25, 0]]
        )
        state = state.replace(
            p_pos=p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 1)
        self.assertEqual(rewards['agent_1'], 1)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        grid = jnp.array([[7, 8, -1], [10, 11, 1]])
        state = state.replace(
            p_pos=p_pos,
            payload=jnp.array([0, -1]),
            grid=grid,
            request=jnp.array([0, 1]),
        )

        obs = self.env.get_obs(state)
        
        # agent 0
        expected_obs = jnp.array([1, 2, 3, 4, 5, 6, 0, 7, 8, -1, 10, 11, 1, 0, 1])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'], expected_obs)
        )

        # agent 1
        expected_obs = jnp.array([4, 5, 6, 1, 2, 3, -1, 7, 8, -1, 10, 11, 1, 0, 1])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'], expected_obs)
        )

    def test_batched_rollout(self):
        self.env = RWARE(
            num_agents=self.num_agents,
            num_cells=self.num_cells,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            }
        )
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)

        # start robots on shelves for easier testing
        p_pos = jnp.concatenate(
            (state.p_pos[:, self.num_agents:self.num_agents+2, ...], state.p_pos[:, self.num_agents:, ...]),
            axis=1
        )
        state = state.replace(p_pos=p_pos)
        initial_state = state

        def get_action(state):
            return {str(f'agent_{i}'): jnp.where(jnp.logical_or(state.step == 1, state.step == 35), 0, 1) for i in range(self.num_agents)}
        
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
                'jaxmarl/environments/marbler/scenarios/test/rware.gif',
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
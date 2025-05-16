"""
Base environment for robotarium simulator
"""

import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Tuple, Optional, Dict

from jaxrobotarium.constants import *
from jaxrobotarium.robotarium_visualizer import *
try:
    # hack for getting around jaxmarl type checking
    from jaxmarl.environments.spaces import Box, Discrete
except Exception as e:
    from jaxrobotarium.spaces import Box, Discrete

from rps_jax.robotarium import *
from rps_jax.robotarium_abc import *
from rps_jax.utilities.controllers import *
from rps_jax.utilities.barrier_certificates2 import *
from rps_jax.utilities.misc import *

@struct.dataclass
class State:
    p_pos: chex.Array = None
    done: chex.Array = None
    step: int = None
    het_rep: chex.Array = None

    # het discovery fields
    landmark_sensed: chex.Array = None
    landmark_tagged: chex.Array = None

    # material transport / warehouse / foraging fields
    zone1_load: int = None
    zone2_load: int = None
    payload: chex.Array = None

    # arctic transport / rware fields
    grid: chex.Array = None

    # rware fields
    request: chex.Array = None

class HetManager:
    def __init__(
            self,
            num_agents,
            type,
            values=None,
            obs_type=None,
            sample=False,
        ):
        """
        Initializes a manager for the heterogeneity representations in the environment.

        Args:
            num_agents: (int) number of agents
            type: (str) type of heterogeneity representation, must be in HET_TYPES defined in constants.py
            sample: (bool) indicates if heterogeneity representation is expected to be resampled at each environment reset
            obs_type: (str) how the observation model represents heterogeneity, if None not represented
        """
        self.num_agents = num_agents
        self.type = type

        # set sampling logic, intended to be used on environment reset()
        if type not in HET_TYPES:
            raise ValueError(f'{type} not in supported heterogeneity types, {HET_TYPES}')
        elif type == 'id':
            # representation is one hot unqiue identifier
            self.representation_set = jnp.eye(num_agents)
            self.sample_fn = lambda key, x, num_agents: x
        elif type == 'class':
            # representation is a one hot class indentifier
            self.representation_set = jnp.array(values)
            if sample == True:
                # TODO: set probabilities per class?
                self.sample_fn = jax.random.choice
            else:
                self.sample_fn = lambda key, x, num_agents: x
        elif type == 'capability_set':
            # representation is a vector of scalar capabilities, sampled from passed in set of possible agents
            self.representation_set = jnp.array(values)
            if sample == True:
                # TODO: set probabilities per class?
                self.sample_fn = jax.random.choice
            else:
                self.sample_fn = lambda key, x, num_agents: x
        elif type == 'capability_dist':
            raise NotImplementedError
        
        def _construct_full_obs(a_idx, state):
            ego_het = state.het_rep[a_idx, :]
            other_het = jnp.roll(state.het_rep, shift=self.num_agents - a_idx - 1, axis=0)[:self.num_agents-1, :]
            return jnp.concatenate([ego_het.flatten(), other_het.flatten()])
        
        # set observation logic, intended to be used in environment get_obs()
        if obs_type is None:
            self.obs_fn = lambda obs, state, a_idx: obs
            self.dim_h = 0
        elif obs_type not in HET_TYPES:
            raise ValueError(f'{type} not in supported heterogeneity types, {HET_TYPES}')
        elif 'id' in type:
            # representation is one hot unqiue identifier
            if 'full' in obs_type:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, _construct_full_obs(a_idx, state)])
                self.dim_h = num_agents * num_agents
            else:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, jnp.eye(num_agents)[a_idx]])
                self.dim_h = num_agents
        elif 'class' in type:
            # representation is a one hot class indentifier
            if 'full' in obs_type:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, _construct_full_obs(a_idx, state)])
                self.dim_h = self.representation_set.shape[-1] * self.num_agents
            else:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, state.het_rep[a_idx]])
                self.dim_h = self.representation_set.shape[-1]
        elif 'capability_set' in obs_type:
            # representation is a vector of scalar capabilities, sampled from passed in set of possible agents
            if 'full' in obs_type:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, _construct_full_obs(a_idx, state)])
                self.dim_h = self.representation_set.shape[-1] * self.num_agents
            else:
                self.obs_fn = lambda obs, state, a_idx: jnp.concatenate([obs, state.het_rep[a_idx]])
                self.dim_h = self.representation_set.shape[-1]
        elif type == 'capability_dist':
            raise NotImplementedError
    
    def sample(self, key):
        """
        Sample a heterogeneity representation from the possible heterogeneity representations

        Args:
            key: (chex.PRNGKey)

        Return:
            (jnp.ndarray) sampled heterogeneity representaiton [num_agents, dim_h]
        """
        idxs = self.sample_fn(key, jnp.arange(self.representation_set.shape[0]), (self.num_agents,))
        return self.representation_set[idxs]

    def process_obs(self, obs, state, a_idx):
        """
        Update observation to include heterogeneity representation

        Args:
            obs: (Dict) original observation to be modified
            state: (State) environment state
            a_idx: (int) index of agent
        
        Returns:
            (Dict) observations with heterogeneity information
        """
        return self.obs_fn(obs, state, a_idx)

class Controller:
    def __init__(
        self,
        controller = None,
        barrier_fn = None,
        **kwargs
    ):
        """
        Initialize wrapper class for handling calling controllers and barrier functions.

        Args:
            controller: (str) name of controller, supported controllers defined in constants.py
            barrier_fn: (str) name of barrier fn, supported barrier functions defined in constants.py 
        """
        if controller is None:
            # if controller is not set, return trivial pass through of actions
            controller = lambda x, g: g
        elif controller not in CONTROLLERS:
            raise ValueError(f'{controller} not in supported controllers, {CONTROLLERS}')
        elif controller == 'si_position':
            controller = create_si_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_position':
            controller = create_clf_unicycle_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_pose':
            controller = create_clf_unicycle_pose_controller(**kwargs.get('controller_args', {}))

        if barrier_fn is None:
            barrier_fn = lambda dxu, x, unused: dxu
        elif barrier_fn not in BARRIERS:
            raise ValueError(f'{controller} not in supported controllers, {CONTROLLERS}')
        elif barrier_fn == 'robust_barriers':
            barrier_fn = create_robust_barriers(safety_radius=SAFETY_RADIUS)

        self.controller = controller
        self.barrier_fn = barrier_fn
    
    def get_action(self, x, g):
        """
        Applies controller and barrier function to get action
        
        Args:
            x: (jnp.ndarray) 3xN states (x, y, theta)
            g: (jnp.ndarray) 2xN (x, y) positions or 3xN poses (x, y, theta)
        
        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        dxu = self.controller(x, g)
        dxu_safe = self.barrier_fn(dxu, x, [])

        return dxu_safe

class RobotariumEnv:
    def __init__(
        self,
        num_agents: int,
        max_steps=MAX_STEPS,
        action_type=DISCRETE_ACT,
        **kwargs
    ) -> None:
        """
        Initialize robotarium environment

        Args:
            num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.max_steps = max_steps
        self.eval = kwargs.get("eval", False)

        # Initialize robotarium and controller backends 
        default_robotarium_args = {'number_of_robots': num_agents, 'show_figure': False, 'sim_in_real_time': False}
        self.robotarium = Robotarium(**kwargs.get('robotarium', default_robotarium_args))
        self.controller = Controller(**kwargs.get('controller', {}))
        self.actuation_noise = kwargs.get('actuation_noise', 0)
        self.step_dist = kwargs.get('step_dist', 0.2)
        self.update_frequency = kwargs.get('update_frequency', 10)

        # Action type
        self.action_dim = 5
        if action_type == DISCRETE_ACT:
            self.action_spaces = {i: Discrete(self.action_dim) for i in self.agents}
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_spaces = {i: Box(0.0, 1.0, (self.action_dim,)) for i in self.agents}
            self.action_decoder = self._decode_continuous_action

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, check.Array], State]) initial observation and environment state
        """

        raise NotImplementedError

    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """
        Environment-specific step transition.
        
        Args:
            key: (chex.PRNGKey)
            state: (State) environment state
            actions: (Dict) agent actions
        
        Returns:
            Tuple(
                (Dict[str, chex.Array]) new observation
                (State) new environment state
                (Dict[str, float]) agent rewards
                (Dict[str, bool]) dones
                (Dict) environment info
            )
        """

        raise NotImplementedError
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards.
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Applies observation function to state.

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]
    
    def _get_violations(self, state: State) -> Dict[str, float]:
        """
        Checks environment for collision and boundary violations.

        Args:
            state: (State) environment state
        
        Returns
            (Dict[str, float]) collision and boundary violations
        """
        b = self.robotarium.boundaries
        p = state.p_pos[:self.num_agents, :].T
        N = self.num_agents

        # Check boundary conditions
        x_out_of_bounds = (p[0, :] < b[0]) | (p[0, :] > (b[0] + b[2]))
        y_out_of_bounds = (p[1, :] < b[1]) | (p[1, :] > (b[1] + b[3]))
        boundary_violations = jnp.where(x_out_of_bounds | y_out_of_bounds, 1, 0)
        boundary_violations = jnp.sum(boundary_violations)

        # Pairwise distance computation for collision checking
        distances = jnp.sqrt(jnp.sum((p[:2, :, None] - p[:2, None, :])**2, axis=0))
        
        collision_matrix = distances < self.robotarium.collision_diameter
        collision_violations = (jnp.sum(collision_matrix) - N) // 2 # Subtract N to remove self-collisions, divide by 2 for symmetry

        return {'collision': collision_violations, 'boundary': boundary_violations}

    def _decode_discrete_action(self, a_idx: int, action: int, state: State):
        """
        Decode action index into null, up, down, left, right actions

        Args:
            a_idx (int): agent index
            action: (int) action index
            state: (State) environment state
        
        Returns:
            (chex.Array) desired (x,y) position
        """
        goals = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        candidate_goals = state.p_pos[a_idx,:2] + (goals[action] * self.step_dist)

        # ensure goals are in bound
        b = jnp.array(self.robotarium.boundaries)
        in_goals = jnp.clip(candidate_goals, b[:2] + 0.1, b[:2] + b[2:] - 0.1)

        return in_goals

    def _decode_continuous_action(self, a_idx: int, action: chex.Array, state: State):
        """
        Trivially returns actions, assumes directly setting v and omega

        Args:
            a_idx: (int) agent index
            action: (chex.Array) action
            state: (State) environment state
        
        Returns:
            (chex.Array) action
        """
        return action
    
    def _noisy_robotarium_step(self, poses: jnp.ndarray, goals: jnp.ndarray, key):
        """
        Wrapper to step robotarium simulator update_frequency times

        Args:
            poses: (jnp.ndarray) Nx3 array of robot poses
            actions: (jnp.ndarray) Nx2 array of robot actions
            key: (chex.PRNGKey) random key for sampling noise
        
        Returns:
            (jnp.ndarray) final poses after update_frequency steps
        """
        poses = poses.T
        goals = goals.T
        orig_dxu = self.controller.get_action(poses, goals) 
        def wrapped_step(poses, unused):
            dxu = jax.lax.cond(
                self.eval,
                lambda _: self.controller.get_action(poses, goals),
                lambda _: orig_dxu,
                operand=None
            )
            _, key_n = jax.random.split(key)
            dxu = dxu + jax.random.normal(key_n, dxu.shape) * self.actuation_noise
            poses = self.robotarium.batch_step(poses, dxu)
            return poses, None
        final_pose, _ = jax.lax.scan(wrapped_step, poses, None, self.update_frequency)

        return final_pose.T
    
    def _robotarium_step(self, poses: jnp.ndarray, goals: jnp.ndarray):
        """
        Wrapper to step robotarium simulator update_frequency times

        Args:
            poses: (jnp.ndarray) Nx3 array of robot poses
            actions: (jnp.ndarray) Nx2 array of robot actions
            update_frequency: (int) number of times to step robotarium simulator
        
        Returns:
            (jnp.ndarray) final poses after update_frequency steps
        """
        poses = poses.T
        goals = goals.T
        orig_dxu = self.controller.get_action(poses, goals) 
        def wrapped_step(poses, unused):
            dxu = jax.lax.cond(
                self.eval,
                lambda _: self.controller.get_action(poses, goals),
                lambda _: orig_dxu,
                operand=None
            )
            poses = self.robotarium.batch_step(poses, dxu)
            return poses, None
        final_pose, _ = jax.lax.scan(wrapped_step, poses, None, self.update_frequency)

        return final_pose.T
    
    #-------------------------------------------------------------
    # Visualization Specific Functions (NOT INTENDED TO BE JITTED)
    #-------------------------------------------------------------
    
    def determine_marker_size(self, marker_size):
        """
        Implementation copied from logic in robotarium_python_simulator/rps/utilities/misc.py

        TODO: move this to rps_jax?
        """

        # Get the x and y dimension of the robotarium figure window in pixels
        fig_dim_pixels = self.visualizer.axes.transData.transform(
            np.array([[self.visualizer.boundaries[2]],[self.visualizer.boundaries[3]]])
        )

        # Determine the ratio of the robot size to the x-axis (the axis are
        # normalized so you could do this with y and figure height as well).
        marker_ratio = (marker_size)/(self.visualizer.boundaries[2])

        # Determine the marker size in points so it fits the window. Note: This is squared
        # as marker sizes are areas.
        return (fig_dim_pixels[0,0] * marker_ratio)**2.

    def render(self, batch_states, seed_index=0, env_index=0):
        """
        Renders rollout from training, designed to work with batched output from training runs

        Args:
            batch_states: (State) rollout states with anticipated dimensions [num_seeds, num_timesteps, num_envs, ...]
            seed_index: (int) seed to visualize
            env_index: (int) env to visualize

        Returns:
            List[Image] rendered rollout frames
        """
        from PIL import Image
        import numpy as np

        # extract poses
        poses = np.array(batch_states.p_pos[seed_index, :-1, env_index, ...].transpose(0, 2, 1)[:, :, :self.num_agents])
        t, _, _ = poses.shape

        # initialize env_state for frame rendering logic
        # TODO: figure out a better way, I kind of hate this
        env_frame = State()
        fields = {}
        for attr in batch_states.__dict__.keys():
            if getattr(batch_states, attr) is None:
                    continue
            fields[f'{attr}'] = getattr(batch_states, attr)[seed_index, 0, env_index, ...]
        env_frame = env_frame.replace(**fields)

        # initialize robotarium visualizer
        self.visualizer = RobotariumVisualizer(self.num_agents, poses[0])

        frames = []
        for i in range(t):
            fields = {}
            for attr in batch_states.__dict__.keys():
                if getattr(batch_states, attr) is None:
                    continue
                fields[f'{attr}'] = getattr(batch_states, attr)[seed_index, i, env_index, ...]
            env_frame = env_frame.replace(**fields)
            self.visualizer.update(poses[i])

            # call scenario specific frame rendering if applicable
            try:
                self.render_frame(env_frame)
            except:
                pass

            # render the current frame
            self.visualizer.figure.canvas.draw()
            frame_image = np.array(self.visualizer.figure.canvas.renderer.buffer_rgba())
            frames.append(Image.fromarray(frame_image))
        
        return frames
    
    def render_frame(self, env_state):
        """
        Scenario specific rendering logic

        Args:
            state: (State) environment state
        """
        raise NotImplementedError        
    
    #-----------------------------------------
    # Deployment Specific Functions
    #-----------------------------------------
    def initial_robotarium_state(self, seed: int = 0):
        """
        Sets initial conditions for robotarium

        Args:
            seed: (int) seed for random functions
        """

        raise NotImplementedError

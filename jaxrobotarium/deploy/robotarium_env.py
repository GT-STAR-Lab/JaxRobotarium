"""
Base robotarium_env implementation compatible with non-Jax Robotarium Python Simulator.
"""

from dataclasses import dataclass
import numpy as jnp
from typing import Tuple, Optional, Dict

from constants import *

from rps.robotarium import *
from rps.robotarium_abc import *
from rps.utilities.controllers import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *

@dataclass
class State:
    p_pos: jnp.ndarray = None
    done: jnp.ndarray = None
    step: int = None
    het_rep: jnp.ndarray = None

    # discovery fields
    landmark_sensed: jnp.ndarray = None
    landmark_tagged: jnp.ndarray = None

    # material transport / warehouse / foraging fields
    zone1_load: int = None
    zone2_load: int = None
    payload: jnp.ndarray = None

    # arctic transport / rware fields
    grid: jnp.ndarray = None

    # rware fields
    request: jnp.ndarray = None

    def replace(self, **kwargs):
        """
        Replace fields in dataclass

        Args:
            kwargs: (Dict) fields to replace
        
        Returns:
            (State) new state with fields replaced
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

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
            self.sample_fn = lambda x, num_agents: x
        elif type == 'class':
            # representation is a one hot class indentifier
            self.representation_set = jnp.array(values)
            if sample == True:
                # TODO: set probabilities per class?
                self.sample_fn = jnp.random.choice
            else:
                self.sample_fn = lambda x, num_agents: x
        elif type == 'capability_set':
            # representation is a vector of scalar capabilities, sampled from passed in set of possible agents
            self.representation_set = jnp.array(values)
            if sample == True:
                # TODO: set probabilities per class?
                self.sample_fn = jnp.random.choice
            else:
                self.sample_fn = lambda x, num_agents: x
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
            key: (chex.PRNGKey) UNUSED HERE
        Return:
            (jnp.ndarray) sampled heterogeneity representaiton [num_agents, dim_h]
        """
        idxs = self.sample_fn(jnp.arange(self.representation_set.shape[0]), (self.num_agents,))
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
            barrier_fn = create_unicycle_differential_drive_barrier_certificate_with_boundary(safety_radius=SAFETY_RADIUS)

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
        dxu_safe = self.barrier_fn(dxu, x, jnp.zeros(0))

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

        # Initialize robotarium and controller backends
        robotarium_args = {
            'number_of_robots': num_agents,
            'show_figure': True,
            'sim_in_real_time': True,
            'initial_conditions': kwargs.get('initial_conditions')
        }
        self.robotarium = Robotarium(**robotarium_args)
        self.controller = Controller(kwargs.get('controller', None), kwargs.get('barrier_fn', None))
        self.step_dist = kwargs.get('step_dist', 0.2)
        self.update_frequency = kwargs.get('update_frequency', 10)

        # Visualizer (trivially set to robotarium so scenario logic remains cross compatible)
        self.visualizer = self.robotarium

        # Action type
        self.action_dim = 5
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action

    def reset(self) -> Tuple[Dict[str, np.ndarray], State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, check.Array], State]) initial observation and environment state
        """

        raise NotImplementedError

    def step_env(
        self, state: State, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], State, Dict[str, float], Dict[str, bool], Dict]:
        """
        Environment-specific step transition.
        
        Args:
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
        Assigns rewards, trivially returns 0 here.
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        raise NotImplementedError
    
    def get_obs(self, state: State) -> Dict[str, np.ndarray]:
        """
        Applies observation function to state.

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        raise NotImplementedError

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

    def _decode_continuous_action(self, a_idx: int, action: np.ndarray, state: State):
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
        dxu = self.controller.get_action(poses, goals)
        for _ in range(self.update_frequency):
            poses = self.robotarium.get_poses()
            dxu = jnp.array(self.controller.get_action(poses, goals))
            self.robotarium.set_velocities(None, dxu)
            self.robotarium.step()
        final_pose = poses.T

        return final_pose
    
    #-----------------------------------------
    # Visualization Specific Functions
    #-----------------------------------------
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

    #-----------------------------------------
    # Deployment Specific Functions
    #-----------------------------------------
    def initial_robotarium_state(self, seed: int = 0):
        """
        Sets initial conditions for robotarium

        Args:
            seed: (int) seed for random functions
        
        Returns:
            (State) initial state
        """

        raise NotImplementedError

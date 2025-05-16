"""
Robots collaborate to unload all material.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class MaterialTransport(RobotariumEnv):
    def __init__(self, num_agents, max_steps=70, **kwargs):
        self.name = 'MARBLER_material_transport'
        self.backend = kwargs.get('backend', 'jax')

        # Heterogeneity
        self.num_sensing = kwargs.get('num_sensing', 2)
        self.num_capturing = kwargs.get('num_capturing', 2)
        default_het_args = {
            'num_agents': num_agents,
            'type': 'capability_set',
            'values': [[.45, 5], [.45, 5], [.15, 15], [.15, 15]],
            'obs_type': None
        }
        het_args = kwargs.get('heterogeneity', default_het_args)
        het_args['num_agents'] = num_agents
        self.het_manager = HetManager(**het_args)

        # Load distribution
        self.zone1_dist = kwargs.get('zone1_dist')
        self.zone2_dist = kwargs.get('zone2_dist')

        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)

        # Reward shaping
        self.load_shaping = kwargs.get('load_shaping', 0.25)
        self.dropoff_shaping = kwargs.get('dropoff_shaping', 0.75)
        self.violation_shaping = kwargs.get('violation_shaping', 0)
        self.time_shaping = kwargs.get('time_shaping', -0.1)

        # Observation space (poses of all agents, zone loads, capabilities)
        self.obs_dim = (3 * self.num_agents) + 2 + self.het_manager.dim_h
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }
        
        # Zone dimensions
        self.dropoff_width = 0.5
        self.zone1_pos = jnp.zeros((2,))  # center of map
        self.zone1_radius = 0.35
        self.zone2_width = 0.5

        # Visualization
        self.robot_markers = []
        self.zone_labels = []
        self.dropoff_marker = None
        self.zone1_marker = None
        self.zone2_marker = None
    
    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # randomly generate initial poses for robots
        key, key_a = jax.random.split(key)
        poses = generate_initial_conditions(
            self.num_agents,
            width=ROBOTARIUM_WIDTH / 4,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
            key=key_a
        )
        self.robotarium.poses = poses

        # get load per zone
        key, key_z1, key_z2 = jax.random.split(key, 3)
        zone1_load = self.zone1_dist['mu'] + self.zone1_dist['sigma'] * jax.random.normal(key_z1, (1,))
        zone2_load = self.zone2_dist['mu'] + self.zone2_dist['sigma'] * jax.random.normal(key_z2, (1,))

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        key, key_het = jax.random.split(key)
        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(key_het),
            zone1_load = zone1_load,
            zone2_load = zone2_load,
            payload = jnp.full((self.num_agents,), 0)
        )

        return self.get_obs(state), state

    def step_env(
        self, key, state: State, actions: Dict
    ) -> Tuple[Dict, State, Dict[str, float], Dict[str, bool], Dict]:
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

        actions = jnp.array([self.action_decoder(i, actions[f'agent_{i}'], state) for i in range(self.num_agents)]).reshape(
            (self.num_agents, -1)
        ) 
        poses = state.p_pos[:self.num_agents, :]

        # update pose
        updated_pose = self._robotarium_step(poses, actions)
        state = state.replace(
            p_pos=jnp.vstack([updated_pose, state.p_pos[self.num_agents:, :]]),
        )

        # check for violations
        violations = self._get_violations(state)

        # get reward
        reward = self.rewards(state)

        # update zone 1 load
        zone1_dist = jnp.linalg.norm(state.p_pos[:, :2] - self.zone1_pos, axis=-1)
        zone1_dist_mask = zone1_dist < self.zone1_radius    # agents in range
        zone1_load_mask = jnp.bitwise_and(jnp.bitwise_and(state.payload == 0, state.zone1_load > 0), zone1_dist_mask)  # agents loading
        zone1_agent_capacity = jnp.sum(jnp.where(zone1_load_mask, state.het_rep[:, 1], 0))  # total capacity of agents loading
        zone1_load = jnp.clip(state.zone1_load - zone1_agent_capacity, 0, jnp.inf)
        state = state.replace(
            zone1_load = zone1_load,
            payload = jnp.logical_or(state.payload, zone1_load_mask)*1
        )

        # update zone 2 load
        bounds = self.robotarium.boundaries # lower left point / width/ height
        zone2_dist_mask = state.p_pos[:, 0] > (bounds[0] + bounds[2] - self.zone2_width)
        zone2_load_mask = jnp.bitwise_and(jnp.bitwise_and(state.payload == 0, state.zone2_load > 0), zone2_dist_mask)
        zone2_agent_capacity = jnp.sum(jnp.where(zone2_load_mask, state.het_rep[:, 1], 0))
        zone2_load = jnp.clip(state.zone2_load - zone2_agent_capacity, 0, jnp.inf)
        state = state.replace(
            zone2_load = zone2_load,
            payload = jnp.logical_or(state.payload, zone2_load_mask)*1
        )

        # update dropoff zone
        dropoff_dist_mask = state.p_pos[:, 0] < (bounds[0] + self.dropoff_width)
        dropoff_mask = jnp.bitwise_and(state.payload > 0, dropoff_dist_mask)
        state = state.replace(
            payload = jnp.where(dropoff_mask, 0, state.payload)
        )
        
        obs = self.get_obs(state)

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        # check if all material has been unloaded
        all_unloaded = jnp.logical_and((state.zone1_load + state.zone2_load) == 0, jnp.all(state.payload == 0))

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full((self.num_agents,), all_unloaded),
            'material_remaining': jnp.full((self.num_agents,), state.zone1_load + state.zone2_load),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

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
        candidate_goals = state.p_pos[a_idx,:2] + (goals[action] * state.het_rep[a_idx, 0])

        # ensure goals are in bound
        b = jnp.array(self.robotarium.boundaries)
        in_goals = jnp.clip(candidate_goals, b[:2] + 0.1, b[:2] + b[2:] - 0.1)

        return in_goals

    def _decode_continuous_action(self, a_idx: int, action, state: State):
        """
        Trivially returns actions, assumes directly setting v and omega

        Args:
            a_idx: (int) agent index
            action: (chex.Array) action
            state: (State) environment state
        
        Returns:
            (chex.Array) action
        """
        return action * state.het_rep[a_idx, 0]

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for loading + shaping reward for unloading + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # update zone 1 load
        zone1_dist = jnp.linalg.norm(state.p_pos[:, :2] - self.zone1_pos, axis=-1)
        zone1_dist_mask = zone1_dist < self.zone1_radius    # agents in range
        zone1_load_mask = jnp.logical_and(jnp.logical_and(state.payload == 0, state.zone1_load > 0), zone1_dist_mask)  # agents loading
        zone1_loaded = jnp.sum(zone1_load_mask*1)

        # update zone 2 load
        bounds = self.robotarium.boundaries # lower left point / width/ height
        zone2_dist_mask = state.p_pos[:, 0] > (bounds[0] + bounds[2] - self.zone2_width)
        zone2_load_mask = jnp.logical_and(jnp.logical_and(state.payload == 0, state.zone2_load > 0), zone2_dist_mask)
        zone2_loaded = jnp.sum(zone2_load_mask*1)

        # update dropoff zone
        dropoff_dist_mask = state.p_pos[:, 0] < (bounds[0] + self.dropoff_width)
        dropoff_mask = jnp.logical_and(state.payload > 0, dropoff_dist_mask)
        dropped_off = jnp.sum(dropoff_mask*1)

        # check if all material unloaded
        all_unloaded = jnp.logical_and((state.zone1_load + state.zone2_load) == 0, jnp.all(state.payload == 0))
        material_remaining = jnp.sum(jnp.where(all_unloaded, 0, 1))

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        rew = (zone1_loaded + zone2_loaded) * self.load_shaping \
            + dropped_off * self.dropoff_shaping \
            + material_remaining * self.time_shaping \
        
        return {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}
    
    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, other_pos, zone loads, het_rep)

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        def _obs(aidx: int):
            """Helper function to create agent observation"""
            
            def shift_array(arr, i):
                """
                Assuming arr is 2D, moves row i to the front
                """
                i = i % arr.shape[0]
                first_part = arr[i:]
                second_part = arr[:i]
                return jnp.concatenate([first_part, second_part])

            # get ego pose and other agent pose
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 3
                other_pos.flatten(),  # num_agents-1, 3
                state.zone1_load,
                state.zone2_load
            ])

            return obs

        return {a: self.het_manager.process_obs(_obs(i), state, i) for i, a in enumerate(self.agents)}
    
    #-----------------------------------------
    # Visualization Specific Functions (NOT INTENDED TO BE JITTED)
    #-----------------------------------------
    
    def render_frame(self, state: State):
        """
        Updates visualizer figure to include goal position markers

        Args:
            state: (State) environment state
        """
        
        # reset markers if at first step
        if state.step == 1:
            self.robot_markers = []
            self.zone_labels = []
            self.dropoff_marker = None
            self.zone1_marker = None
            self.zone2_marker = None
        
        # add markers for robots, wider is larger load
        poses = state.p_pos
        if not self.robot_markers:
            # green for sensing
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(poses[i, 0]),
                    jnp.array(poses[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(state.het_rep[i, 1] * 0.02),
                    facecolors='none',
                    edgecolors='black',
                    linewidth=3
                ) for i in range(self.num_agents)
            ]
        
        # add zones
        if not self.dropoff_marker:
            self.dropoff_marker = self.visualizer.axes.add_patch(
                patches.Rectangle([-1.5, -1], self.dropoff_width, 2, color='purple', zorder=-2)
            )
        if not self.zone1_marker:
            self.zone1_marker = self.visualizer.axes.scatter(
                    0, 0, s=self.determine_marker_size(self.zone1_radius),
                    marker='o', facecolors='none', edgecolors='orange', linewidth=3, zorder=-2
            )
        if not self.zone2_marker:
            self.zone2_marker = self.visualizer.axes.add_patch(
                patches.Rectangle([1.5 - self.zone2_width, -1], self.zone2_width , 2, color='blue', zorder=-2)
            )

        # add labels
        if not self.zone_labels:
            self.zone_labels.append(
                self.visualizer.axes.text(0, 0, jnp.round(state.zone1_load, 2), verticalalignment='center', horizontalalignment='center')
            )
            self.zone_labels.append(
                self.visualizer.axes.text(1.5 - self.zone2_width/2, 0, jnp.round(state.zone2_load, 2), verticalalignment='center', horizontalalignment='center')
            )

        
        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(poses[i, :2])
            self.robot_markers[i].set_edgecolor('green' if state.payload[i] else 'black')
        
        # update labels
        self.zone_labels[0].set_text(jnp.round(state.zone1_load, 2))
        self.zone_labels[1].set_text(jnp.round(state.zone2_load, 2))


    #-----------------------------------------
    # Deployment Specific Functions
    #-----------------------------------------
    def initialize_robotarium_state(self, seed: int = 0):
        """
        Sets initial conditions for robotarium

        Args:
            seed: (int) seed for random functions
        
        Returns:
            (jnp.ndarray) initial poses (3xN) for robots
        """

        # randomly generate initial poses for robots
        poses = generate_initial_conditions(
            self.num_agents,
            width=ROBOTARIUM_WIDTH / 4,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
        )

        zone1_load = self.zone1_dist['mu'] + self.zone1_dist['sigma'] * jnp.random.normal(size=(1,))
        zone2_load = self.zone2_dist['mu'] + self.zone2_dist['sigma'] * jnp.random.normal(size=(1,))

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(None),
            zone1_load = zone1_load,
            zone2_load = zone2_load,
            payload = jnp.full((self.num_agents,), 0)
        )

        return state
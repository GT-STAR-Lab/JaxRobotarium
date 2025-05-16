"""
Robots collaborate to complete the maximum number of deliveries.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class Warehouse(RobotariumEnv):
    def __init__(self, num_agents, max_steps=70, **kwargs):
        self.name = 'warehouse'
        self.backend = kwargs.get('backend', 'jax')

        # Heterogeneity
        default_het_args = {
            'num_agents': num_agents,
            'type': 'class',
            'values': [[1, 0], [1, 0], [0, 1], [0, 1]],
            'obs_type': None
        }
        het_args = kwargs.get('heterogeneity', default_het_args)
        het_args['num_agents'] = num_agents
        self.het_manager = HetManager(**het_args)

        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)

        # Reward shaping
        self.load_shaping = kwargs.get('load_shaping', 1)
        self.dropoff_shaping = kwargs.get('dropoff_shaping', 3)
        self.violation_shaping = kwargs.get('violation_shaping', 0)

        # Observation space (poses of all agents, heterogeneity)
        self.obs_dim = (3 * self.num_agents) + self.het_manager.dim_h
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }
        
        # zone info
        self.zone_width = 0.5

        # Visualization
        self.robot_markers = []
        self.zone_markers = []
    
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
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
            key=key_a
        )
        self.robotarium.poses = poses

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        key, key_het = jax.random.split(key)
        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(key_het),
            payload = jnp.full((self.num_agents,), 0),
            zone1_load = 0,
            zone2_load = 0
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

        # unload
        bounds = self.robotarium.boundaries # lower left point / width / height
        able_to_unload = jnp.bitwise_and(state.p_pos[:, 0] < (bounds[0] + self.zone_width), state.payload > 0)
        green_unload = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] > 0, state.het_rep[:,0]), able_to_unload)
        red_unload = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] < 0, state.het_rep[:,1]), able_to_unload)
        payload = jnp.where(jnp.logical_or(green_unload, red_unload), 0, state.payload)
        green_deliveries = jnp.sum(green_unload*1)
        red_deliveries = jnp.sum(red_unload*1)
        state = state.replace(
            zone1_load=state.zone1_load + green_deliveries,
            zone2_load=state.zone2_load + red_deliveries
        )

        # load
        bounds = self.robotarium.boundaries # lower left point / width / height
        able_to_load = jnp.bitwise_and(state.p_pos[:, 0] > (bounds[0] + bounds[2] - self.zone_width), state.payload == 0)
        green_load = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] < 0, state.het_rep[:,0]), able_to_load)
        red_load = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] > 0, state.het_rep[:,1]), able_to_load)
        payload = jnp.where(jnp.logical_or(red_load, green_load), 1, payload)

        # update payload
        state = state.replace(payload=payload)
        
        obs = self.get_obs(state)

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'deliveries_made': jnp.full((self.num_agents,), state.zone1_load + state.zone2_load),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for loading + shaping reward for unloading + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # unload
        bounds = self.robotarium.boundaries # lower left point / width / height
        able_to_unload = jnp.bitwise_and(state.p_pos[:, 0] < (bounds[0] + self.zone_width), state.payload > 0)
        green_unload = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] > 0, state.het_rep[:,0]), able_to_unload)
        red_unload = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] < 0, state.het_rep[:,1]), able_to_unload)
        green_deliveries = jnp.sum(green_unload*1)
        red_deliveries = jnp.sum(red_unload*1)

        # load
        bounds = self.robotarium.boundaries # lower left point / width / height
        able_to_load = jnp.bitwise_and(state.p_pos[:, 0] > (bounds[0] + bounds[2] - self.zone_width), state.payload == 0)
        green_load = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] < 0, state.het_rep[:,0]), able_to_load)
        red_load = jnp.bitwise_and(jnp.bitwise_and(state.p_pos[:, 1] > 0, state.het_rep[:,1]), able_to_load)
        green_loaded = jnp.sum(green_load*1)
        red_loaded = jnp.sum(red_load*1)

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        rew = (green_loaded + red_loaded) * self.load_shaping \
            + (green_deliveries + red_deliveries) * self.dropoff_shaping \
        
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
            self.zone_markers = []
        
        # add markers for robots, wider is larger load
        poses = state.p_pos
        if not self.robot_markers:
            # green for sensing
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(poses[i, 0]),
                    jnp.array(poses[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(0.15),
                    facecolors='none',
                    edgecolors='green' if state.het_rep[i, 0] else 'red',
                    linewidth=3
                ) for i in range(self.num_agents)
            ]
        
        # add zones
        if not self.zone_markers:
            self.zone_markers.append(self.visualizer.axes.add_patch(
                patches.Rectangle([-1.5, 0], self.zone_width, 1, color='green', zorder=-2)
            ))
            self.zone_markers.append(self.visualizer.axes.add_patch(
                patches.Rectangle([-1.5, -1], self.zone_width, 1, color='red', zorder=-2)
            ))
            self.zone_markers.append(self.visualizer.axes.add_patch(
                patches.Rectangle([1.5-self.zone_width, 0], self.zone_width, 1, color='red', zorder=-2)
            ))
            self.zone_markers.append(self.visualizer.axes.add_patch(
                patches.Rectangle([1.5-self.zone_width, -1], self.zone_width, 1, color='green', zorder=-2)
            ))
        
        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(poses[i, :2])
            self.robot_markers[i].set_facecolor('gray' if state.payload[i] else 'none')


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
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
        )

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(None),
            payload = jnp.full((self.num_agents,), 0),
            zone1_load = 0,
            zone2_load = 0
        )

        return state

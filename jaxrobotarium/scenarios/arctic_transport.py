"""
Robots collaborate to traverse arctic terrain.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class ArcticTransport(RobotariumEnv):
    def __init__(self, num_agents, max_steps=80, **kwargs):
        self.name = 'arctic_transport'
        self.backend = kwargs.get('backend', 'jax')

        # Arctic transport specific constraint at the moment
        assert(num_agents == 4), "Arctic Transport requires 4 agents"

        # Heterogeneity
        default_het_args = {
            'num_agents': num_agents,
            'type': 'class',
            'values': [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'obs_type': None
        }
        het_args = kwargs.get('heterogeneity', default_het_args)
        het_args['num_agents'] = num_agents
        self.het_manager = HetManager(**het_args)

        # Initialize backend
        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)

        # Reward shaping
        self.dist_shaping = kwargs.get('dist_shaping', -0.05)
        self.violation_shaping = kwargs.get('violation_shaping', 0)
        self.time_shaping = kwargs.get('time_shaping', -0.1)

        # Observation space (poses of all agents, agent_terrain type, drone observations, heterogeneity)
        self.obs_dim = (3 * self.num_agents) + 1 + (9 * 2) + self.het_manager.dim_h
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }

        # Visualization
        self.robot_markers = []
        self.terrain_markers = []

    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # starting poses are fixed
        poses = jnp.array([
            [-0.3, -0.8, 0.0],
            [0.3, -0.8, 0.0],
            [-0.9, -0.8, 0.0],
            [0.9, -0.8, 0.0],
        ]).T
        self.robotarium.poses = poses[:, :self.num_agents]

        # randomly generate terrain grid
        # 0 is normal terrain
        # 1 is ice
        # 2 is water
        # 3 is the goal
        key, key_t = jax.random.split(key)
        terrain_grid = jax.random.randint(key_t, (6, 12), 0, 3)
        terrain_grid = jnp.concatenate(
            [   
                jnp.ones((1,12)) * 3, # goal
                terrain_grid, 
                jnp.zeros((1, 12)) # start
            ],
            axis=0
        )
        
        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        key, key_het = jax.random.split(key)
        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep=self.het_manager.sample(key_het),
            grid=terrain_grid,
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

        obs = self.get_obs(state)

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        # check if all agents on goal
        bounds = jnp.array(self.robotarium.boundaries)
        all_reached = jnp.all(state.p_pos[2:self.num_agents, 1] > (bounds[3] / 8)*3)

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full((self.num_agents,), all_reached),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def _cell_from_pos(self, pos):
        """
        Convert a position to a cell index

        Args:
            pos: (chex.Array) position
        Returns:
            (chex.Array) cell index
        """

        cell_y = jnp.clip(-(pos[1] - 1) / .25, 0, 7)
        cell_x = jnp.clip((pos[0] + 1.5) / .25, 0, 11)
        
        return jnp.array([cell_y, cell_x], dtype=jnp.int32)
    
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

        # get terrain and agent type
        terrain_type = state.grid[
            self._cell_from_pos(state.p_pos[a_idx, :2])[0],
            self._cell_from_pos(state.p_pos[a_idx, :2])[1]
        ]
        agent_type = jnp.argmax(state.het_rep[a_idx])

        # set step size
        step = jnp.where(agent_type == terrain_type, 0.3, 0.1)
        step = jnp.where(jnp.logical_and(agent_type != 0, terrain_type % 3 == 0), 0.2, step)
        step = jnp.where(agent_type == 0, 0.2, step)

        candidate_goals = state.p_pos[a_idx,:2] + (goals[action] * step)

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
        # get terrain and agent type
        terrain_type = state.grid[self._cell_from_pos(state.p_pos[a_idx, :2])]
        agent_type = jnp.argmax(state.het_rep[a_idx])

        # set step size
        step = jnp.where(agent_type == terrain_type, 0.3, 0.1)
        step = jnp.where(jnp.logical_and(agent_type != 0, terrain_type % 3 == 0), 0.2, step)
        step = jnp.where(agent_type == 0, 0.2, step)

        return action * step
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for sensing + shaping reward for capture + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # check non-drone distance from goal
        bounds = jnp.array(self.robotarium.boundaries)
        dist = state.p_pos[2:self.num_agents, 1] - (bounds[3] / 8)*3
        dist_rew =  jnp.abs(jnp.sum(jnp.where(dist > 0, 0, dist)))

        # check for all agents on goal
        all_reached = jnp.all(state.p_pos[2:self.num_agents, 1] > (bounds[3] / 8)*3)

        # compute task reward
        rew = (dist_rew * self.dist_shaping) + (~all_reached * self.time_shaping)

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        return {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}

    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, other_pos, prey_pos, het_rep)

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        agent_cell_types = jnp.array([
            state.grid[self._cell_from_pos(state.p_pos[i, :2])[0], self._cell_from_pos(state.p_pos[i, :2])[1]] \
            for i in range(self.num_agents)
        ])

        # get drone observations
        padded_grid = jnp.pad(state.grid, ((1, 1), (1, 1)), mode='constant', constant_values=-1)
        drone1_cell = self._cell_from_pos(state.p_pos[0, :2])
        x_min, y_min = drone1_cell[0], drone1_cell[1]
        drone1_obs = jax.lax.dynamic_slice(padded_grid, (x_min, y_min), (3, 3))  # get drone1 observation
        drone2_cell = self._cell_from_pos(state.p_pos[1, :2])
        x_min, y_min = drone2_cell[0], drone2_cell[1]
        drone2_obs = jax.lax.dynamic_slice(padded_grid, (x_min, y_min), (3, 3))  # get drone2 observation
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
                agent_cell_types[aidx].reshape(1,),  # 1
                drone1_obs.flatten(),  # 3, 3
                drone2_obs.flatten(),  # 3, 3
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
            self.terrain_markers = []
        
        # add markers for robots
        poses = state.p_pos[:self.num_agents, :2]
        robot_colors = ['black', 'black', 'blue', 'cyan']
        terrain_colors = ['white', 'blue', 'cyan', 'green']
        if not self.robot_markers:
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(poses[i, 0]),
                    jnp.array(poses[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(0.15),
                    facecolors='none',
                    edgecolors=robot_colors[i],
                    linewidth=3
                ) for i in range(self.num_agents)
            ]
        
        if not self.terrain_markers:
            for i in range(8):
                for j in range(12):
                    terrain_type = state.grid[i, j]
                    color = terrain_colors[int(terrain_type)]
                    self.terrain_markers.append(
                        self.visualizer.axes.add_patch(
                            patches.Rectangle(
                                (j*0.25-1.5, (-i*0.25)+0.75),
                                0.25,
                                0.25,
                                color=color,
                                alpha=0.5,
                                zorder=-1
                            )
                        )
                    )
        
        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(poses[i, :2])


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

        # starting poses are fixed
        poses = jnp.array([
            [-0.3, -0.8, 0.0],
            [0.3, -0.8, 0.0],
            [-0.9, -0.8, 0.0],
            [0.9, -0.8, 0.0],
        ])

        terrain_grid = jnp.random.randint(0, 3, size=(6,12))
        terrain_grid = jnp.concatenate(
            [   
                jnp.ones((1,12)) * 3, # goal
                terrain_grid, 
                jnp.zeros((1, 12)) # start
            ],
            axis=0
        )

        state = State(
            p_pos=poses,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(None),
            grid = terrain_grid
        )

        return state

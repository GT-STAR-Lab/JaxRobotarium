"""
Discovery where sensing robots discover landmarks and tagging robots tag landmarks.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class Discovery(RobotariumEnv):
    def __init__(self, num_agents, max_steps=80, **kwargs):
        self.name = 'MARBLER_het_discovery'
        self.backend = kwargs.get('backend', 'jax')

        self.num_landmarks = kwargs.get('num_landmarks', 6)

        # Heterogeneity
        self.num_sensing = kwargs.get('num_sensing', 2)
        self.num_tagging = kwargs.get('num_tagging', 2)
        default_het_args = {
            'num_agents': num_agents,
            'type': 'capability_set',
            'values': [[0.45, 0], [0.45, 0], [0, 0.25], [0, 0.25]],
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
        self.sense_shaping = kwargs.get('sense_shaping', 1)
        self.tag_shaping = kwargs.get('tag_shaping', 5)
        self.violation_shaping = kwargs.get('violation_shaping', 0)
        self.time_shaping = kwargs.get('time_shaping', -0.05)

        # Observation space (poses of all agents, landmark locations if sensed, capabilities)
        self.obs_dim = (3 * self.num_agents) + (3 * self.num_landmarks) + self.het_manager.dim_h
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }

        # Visualization
        self.robot_markers = []
        self.landmark_markers = []

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
        agent_poses = generate_initial_conditions(
            self.num_agents,
            width=ROBOTARIUM_WIDTH / 3,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
            key=key_a
        )
        self.robotarium.poses = agent_poses[:, :self.num_agents]

        # randomly generate initial poses for landmarks
        key, key_l = jax.random.split(key)
        landmark_poses = generate_initial_conditions(
            self.num_landmarks,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
            key=key_l
        )
        
        poses = jnp.concatenate([agent_poses, landmark_poses], axis=-1)

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        key, key_het = jax.random.split(key)
        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(key_het),
            landmark_sensed = jnp.full((self.num_landmarks,), False),
            landmark_tagged = jnp.full((self.num_landmarks,), False)
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

        # update sensed landmarks (if landmark in range, mark as sensed if not already marked)
        landmark_pos = state.p_pos[self.num_agents:, :2]    # get x, y of landmarks
        sense_pos = state.p_pos[:self.num_sensing, :2]  # get x, y of only sensing agents
        sense_dist = jnp.linalg.norm(sense_pos[:, None] - landmark_pos[None, :], axis=-1)    # get dist from all sensing agents to all landmarks
        sensed = sense_dist < state.het_rep[:self.num_sensing, 0, None] # compare to sensing radii per agent

        # update tagged landmarks (if landmark tagged, mark as tagged if not already tagged)
        tag_pos = state.p_pos[self.num_sensing:self.num_sensing+self.num_tagging, :2]  # get x, y of only tagging agents
        tag_dist = jnp.linalg.norm(tag_pos[:, None] - landmark_pos[None, :], axis=-1)    # get dist from all tagging agents to all landmarks
        tagged = tag_dist < state.het_rep[self.num_sensing:self.num_sensing+self.num_tagging, 1, None] # compare to tag radii per agent

        state = state.replace(landmark_sensed=jnp.logical_or(state.landmark_sensed, sensed.any(axis=0)))
        state = state.replace(landmark_tagged=jnp.logical_or(state.landmark_tagged, tagged.any(axis=0)))

        obs = self.get_obs(state)

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        # check if all landmarks tagged
        all_tagged = jnp.all(state.landmark_tagged)

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full((self.num_agents,), all_tagged),
            'landmark_sensed': jnp.full((self.num_agents,), jnp.sum(state.landmark_sensed)),
            'landmark_tagged': jnp.full((self.num_agents,), jnp.sum(state.landmark_tagged)),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for sensing + shaping reward for tag + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # get sensed landmarks (if landmark in range)
        landmark_pos = state.p_pos[self.num_agents:, :2]    # get x, y of landmarks
        sense_pos = state.p_pos[:self.num_sensing, :2]  # get x, y of only sensing agents
        sense_dist = jnp.linalg.norm(sense_pos[:, None] - landmark_pos[None, :], axis=-1)    # get dist from all sensing agents to all landmarks
        sensed = sense_dist < state.het_rep[:self.num_sensing, 0, None] # compare to sensing radii per agent
        sensed = sensed.any(axis=0)
        sensed = jnp.logical_or(state.landmark_sensed, sensed)
        num_sensed = jnp.sum(sensed*1 - state.landmark_sensed*1) # multiplied by 1 to get conversion to int

        # update tagged landmarks (if landmark in range)
        tag_pos = state.p_pos[self.num_sensing:self.num_sensing+self.num_tagging, :2]  # get x, y of only landmark agents
        landmark_dist = jnp.linalg.norm(tag_pos[:, None] - landmark_pos[None, :], axis=-1)    # get dist from all tagging agents to all landmarks
        tagged = landmark_dist < state.het_rep[self.num_sensing:self.num_sensing+self.num_tagging, 1, None] # compare to tag radii per agent
        tagged = jnp.logical_or(state.landmark_tagged, tagged)
        num_tagged = jnp.sum(tagged*1 - state.landmark_tagged*1) # multiplied by 1 to get conversion to int

        # check if all landmarks tagged, if so don't apply penalty
        all_tagged = jnp.sum(tagged) == self.num_landmarks
        landmark_remaining = jnp.where(all_tagged, 0, 1)

        # compute task reward
        rew = (num_sensed * self.sense_shaping) + (num_tagged * self.tag_shaping) + (landmark_remaining * self.time_shaping)

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        return {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}

    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, other_pos, landmark_pos, het_rep)

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

            # get location of landmark if sensed
            landmark_pos = jnp.where(state.landmark_sensed[:, None], state.p_pos[self.num_agents:, :], -5.0)

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 3
                other_pos.flatten(),  # num_agents-1, 3
                landmark_pos.flatten(), # num_landmarks, 3
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
            self.landmark_markers = []
            self.robot_markers = []
        
        landmarks = state.p_pos[self.num_agents:, :2]
        sensing = state.p_pos[:self.num_sensing, :2]
        tagging = state.p_pos[self.num_sensing:self.num_sensing+self.num_tagging, :2]

        # add markers for landmarks        
        if not self.landmark_markers:
            self.landmark_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(landmarks[i, 0]),
                    jnp.array(landmarks[i, 1]),
                    marker='.',
                    s=self.determine_marker_size(.05),
                    facecolors='black',
                    zorder=-2
                ) for i in range(self.num_landmarks)
            ]
        
        # add markers for robots
        if not self.robot_markers:
            # green for sensing
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(sensing[i, 0]),
                    jnp.array(sensing[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(state.het_rep[i, 0]),
                    facecolors='none',
                    edgecolors='green',
                    zorder=-2,
                    linewidth=3
                ) for i in range(self.num_sensing)
            ]

            # blue for tagging
            self.robot_markers += [
                self.visualizer.axes.scatter(
                    jnp.array(tagging[i, 0]),
                    jnp.array(tagging[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(state.het_rep[i+self.num_sensing, 1]),
                    facecolors='none',
                    edgecolors='blue',
                    zorder=-2,
                    linewidth=3
                ) for i in range(self.num_tagging)
            ]
        
        # update robot marker positions
        for i in range(self.num_sensing):
            self.robot_markers[i].set_offsets(sensing[i])
        for i in range(self.num_tagging):
            self.robot_markers[i+self.num_sensing].set_offsets(tagging[i])
        
        # update landmark markers
        for i in range(self.num_landmarks):
            if state.landmark_sensed[i]:
                self.landmark_markers[i].set_facecolor('green')
            if state.landmark_tagged[i]:
                self.landmark_markers[i].set_sizes([0, 0])


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
        agent_poses = generate_initial_conditions(
            self.num_agents,
            width=ROBOTARIUM_WIDTH / 3,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
        )

        # randomly generate initial poses for landmarks
        landmark_poses = generate_initial_conditions(
            self.num_landmarks,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
        )
        
        poses = jnp.concatenate([agent_poses, landmark_poses], axis=-1)

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep = self.het_manager.sample(None),
            landmark_sensed = jnp.full((self.num_landmarks,), False),
            landmark_tagged = jnp.full((self.num_landmarks,), False)
        )

        return state

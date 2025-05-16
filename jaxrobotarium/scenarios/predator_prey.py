"""
Predator Prey where predator agents must collaborate to tag a more agile prey.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class PredatorPrey(RobotariumEnv):
    def __init__(self, num_agents, max_steps=80, **kwargs):
        self.name = 'predator_prey'
        self.backend = kwargs.get('backend', 'jax')

        # Predator tag radius
        self.tag_radius = kwargs.get('tag_radius', 0.2)
        self.prey_step = kwargs.get('prey_step', 0.3)

        # Initialize backend
        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)

        # Reward shaping
        self.tag_shaping = kwargs.get('tag_shaping', 10)
        self.violation_shaping = kwargs.get('violation_shaping', 0)
        self.time_shaping = kwargs.get('time_shaping', 0)

        # Observation space (poses of all agents, prey pose)
        self.obs_dim = 3 * (self.num_agents + 1)
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }

        # Visualization
        self.robot_markers = []
        self.prey_marker = None
        self.prev_tag_count = 0

    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # randomly generate initial poses for robots and prey
        key, key_a = jax.random.split(key)
        poses = generate_initial_conditions(
            self.num_agents + 1,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
            key=key_a
        )
        self.robotarium.poses = poses[:, :self.num_agents]

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            landmark_tagged = jnp.full((1,), 0) # track number of times prey is tagged
        )

        return self.get_obs(state), state
    
    def _prey_policy(self, state: State) -> jnp.ndarray:
        """
        Move the prey based on heuristic in FACMAC.
        This version samples both directions and step sizes, and selects the candidate
        position that maximizes distance to the nearest predator.

        USED CHATGPT FOR THIS

        Args:
            state: (State) environment state
        Returns:
            (jnp.ndarray) new prey position
        """
        prey_pos = state.p_pos[self.num_agents, :2]
        predator_pos = state.p_pos[:self.num_agents, :2]

        num_angles = 8
        num_steps = 4
        max_step = self.prey_step

        # directions
        angles = jnp.linspace(0, 2 * jnp.pi, num_angles, endpoint=False)
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (num_angles, 2)

        # step sizes
        step_sizes = jnp.linspace(0.05, max_step, num_steps)  # (num_steps,)

        # directions and step sizes -> (num_angles * num_steps, 2)
        directions = directions[:, None, :]  # (num_angles, 1, 2)
        step_sizes = step_sizes[None, :, None]  # (1, num_steps, 1)
        displacements = directions * step_sizes  # (num_angles, num_steps, 2)
        displacements = displacements.reshape(-1, 2)  # (num_candidates, 2)

        candidates = prey_pos + displacements  # (num_candidates, 2)

        # clip candidates to be within the bounds of the robotarium
        bounds = self.robotarium.boundaries
        candidates_x = jnp.clip(candidates[:, 0], bounds[0] + 0.1, bounds[0] + bounds[2] - 0.1)
        candidates_y = jnp.clip(candidates[:, 1], bounds[1] + 0.1, bounds[1] + bounds[3] - 0.1)
        candidates = jnp.stack([candidates_x, candidates_y], axis=-1)  # (num_candidates, 2)

        # pairwise distances between candidates and predators
        # candidates: (num_candidates, 2), predator_pos: (num_predators, 2)
        diff = candidates[:, None, :] - predator_pos[None, :, :]  # (num_candidates, num_predators, 2)
        dists = jnp.linalg.norm(diff, axis=-1)  # (num_candidates, num_predators)

        # for each candidate, find distance to closest predator
        min_dists = jnp.min(dists, axis=1)  # (num_candidates,)

        # get the candidate with the largest min-distance
        best_idx = jnp.argmax(min_dists)
        best_pos = candidates[best_idx]
        best_pos = jnp.concatenate([best_pos, jnp.array([0.0])])  # add orientation, doesn't matter since we assume holonomic

        return best_pos

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

        updated_prey_pose = self._prey_policy(state)

        actions = jnp.array([self.action_decoder(i, actions[f'agent_{i}'], state) for i in range(self.num_agents)]).reshape(
            (self.num_agents, -1)
        ) 
        poses = state.p_pos[:self.num_agents, :]

        # update pose
        updated_pose = self._robotarium_step(poses, actions)
        state = state.replace(
            p_pos=jnp.vstack([updated_pose, updated_prey_pose]),
        )

        # check for violations
        violations = self._get_violations(state)

        # get reward
        reward = self.rewards(state)

        # update tagged state
        agent_pos = state.p_pos[:self.num_agents, :2]  # get x, y of only tagging agents
        dist = jnp.linalg.norm(agent_pos - state.p_pos[self.num_agents, :2], axis=-1)    # get dist from all tagging agents to prey
        tagged = dist < self.tag_radius # compare to tag radius.

        # update tag count
        state = state.replace(landmark_tagged=state.landmark_tagged + tagged.any()*1) # multiplied by 1 to get conversion to int

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
            'prey_tagged': jnp.full((self.num_agents,), jnp.sum(state.landmark_tagged)),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for tag + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # check if prey tagged
        agent_pos = state.p_pos[:self.num_agents, :2]  # get x, y of only tagging agents
        dist = jnp.linalg.norm(agent_pos - state.p_pos[self.num_agents, :2], axis=-1)    # get dist from all tagging agents to prey
        tagged = (dist < self.tag_radius).any()*1 # compare to tag radius

        # compute task reward
        rew = tagged * self.tag_shaping

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        return {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}

    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, other_pos, prey_pos)

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

            # get location of prey
            prey_pos = state.p_pos[self.num_agents, :]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 3
                other_pos.flatten(),  # num_agents-1, 3
                prey_pos.flatten(), # num_landmarks, 3
            ])

            return obs

        return {a: _obs(i) for i, a in enumerate(self.agents)}
    
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
            self.prey_marker = None
            self.robot_markers = []
            self.prev_tag_count = 0
        
        robots = state.p_pos[:self.num_agents, :2]
        prey = state.p_pos[self.num_agents, :2].flatten()

        # add marker for prey     
        if not self.prey_marker:
            self.prey_marker = self.visualizer.axes.scatter(
                jnp.array(prey[0]),
                jnp.array(prey[1]),
                marker='.',
                s=self.determine_marker_size(.15),
                facecolors='green',
                zorder=-2
        )
        
        # add markers for robots
        if not self.robot_markers:
            # green for sensing
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(robots[i, 0]),
                    jnp.array(robots[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(self.tag_radius),
                    facecolors='none',
                    edgecolors='black',
                    zorder=-2,
                    linewidth=3
                ) for i in range(self.num_agents)
            ]
        
        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(robots[i])
        
        # update prey marker position
        self.prey_marker.set_offsets(prey)
        
        # if tag count grew, update prey marker color
        if state.landmark_tagged > self.prev_tag_count:
            self.prey_marker.set_facecolor('red')
            self.prev_tag_count = state.landmark_tagged
        else:
            self.prey_marker.set_facecolor('green')

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

        # randomly generate initial poses for robots and prey
        poses = generate_initial_conditions(
            self.num_agents+1,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
        )
        
        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            landmark_tagged = jnp.full((1,), 0) # track number of times prey is tagged
        )

        return state

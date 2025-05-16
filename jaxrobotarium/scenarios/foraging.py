"""
Robots collaborate to forage resources for the team.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *


class Foraging(RobotariumEnv):
    def __init__(self, num_agents, max_steps=70, **kwargs):
        self.name = 'MARBLER_foraging'
        self.backend = kwargs.get('backend', 'jax')

        self.forage_radius = 0.3
        self.num_resources = kwargs.get('num_resources', 2)

        # Heterogeneity
        default_het_args = {
            'num_agents': num_agents,
            'type': 'capability_set',
            'values': [[1], [2]],
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
        self.forage_shaping = kwargs.get('forage_shaping', 1)
        self.time_shaping = kwargs.get('time_shaping', -0.01)
        self.violation_shaping = kwargs.get('violation_shaping', 0)

        # Observation space (poses of all agents, poses and level of all food, heterogeneity)
        self.obs_dim = (3*self.num_agents) + (3*self.num_resources) + self.het_manager.dim_h

        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }
        
        # Visualization
        self.robot_markers = []
        self.resource_markers = []
        self.labels = []

    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # randomly generate initial poses for robots and resources
        key, key_p = jax.random.split(key)
        poses = generate_initial_conditions(
            self.num_agents + self.num_resources,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
            key=key_p
        )
        self.robotarium.poses = poses[:, :self.num_agents]

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        # sample agent levels
        key, key_het = jax.random.split(key)
        het_rep = self.het_manager.sample(key_het)

        # set resource levels based on random subsets of the agent levels
        key, key_r = jax.random.split(key)
        levels = het_rep.flatten()
        level_permutations = jnp.array([levels for _ in range(self.num_resources)])
        resource_mask = jax.random.permutation(key_r, jnp.triu(jnp.ones((self.num_resources, self.num_agents))), axis=0)
        resource_levels = jnp.sum(level_permutations * resource_mask, axis=-1).flatten()

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep=het_rep,
            payload=resource_levels # use payload here to track food levels
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

        # update foraged resources
        resource_pos = state.p_pos[self.num_agents:, :2]    # get x, y of resources
        agent_pos = state.p_pos[:self.num_agents, :2]  # get x, y of  agents
        dist = jnp.linalg.norm(resource_pos[:, None] - agent_pos[None, :], axis=-1)  # get dist from all agents to all resources
        forage_dist = dist < self.forage_radius
        agent_levels = jnp.array([state.het_rep.flatten() for _ in range(self.num_resources)])
        forage_levels = jnp.sum(jnp.where(forage_dist, agent_levels, 0), axis=-1).flatten()
        foraged = jnp.where(forage_levels >= state.payload, -1, state.payload)*1. # set foraged resources to -1 to indicate they have been collected
        state = state.replace(payload=foraged)

        obs = self.get_obs(state)

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        # check if all resources foraged
        all_foraged = jnp.all(foraged < 0)
        num_foraged = jnp.sum(jnp.where(foraged < 0, 1, 0))

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full((self.num_agents,), all_foraged),
            'resources_foraged': jnp.full((self.num_agents,), num_foraged)
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward for foraging + time penalty + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        resource_pos = state.p_pos[self.num_agents:, :2]    # get x, y of resources
        agent_pos = state.p_pos[:self.num_agents, :2]  # get x, y of  agents
        dist = jnp.linalg.norm(resource_pos[:, None] - agent_pos[None, :], axis=-1)  # get dist from all agents to all resources
        forage_dist = dist < self.forage_radius
        agent_levels = jnp.array([state.het_rep.flatten() for i in range(self.num_resources)])
        forage_levels = jnp.sum(jnp.where(forage_dist, agent_levels, 0), axis=-1).flatten()
        foraged = jnp.where(forage_levels >= state.payload, 0, state.payload) # set foraged resources this step to 0
        level_foraged = jnp.sum(jnp.where(state.payload < 0, 0, state.payload)- foraged)
        all_foraged = jnp.all(foraged < 0)*1 # convert to int

        # compute task reward
        rew = (level_foraged * self.forage_shaping) + (all_foraged * self.time_shaping)

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        return {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}
    
    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, other_pos, forage_pos || forage_level, het_rep)

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

            # get ego pose and other agent poses
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]

            # get location of resources
            resource_pos = state.p_pos[self.num_agents:, :2]

            # get level of resources
            resource_level = state.payload

            resource_info = jnp.concatenate([resource_pos, resource_level.reshape(-1, 1)], axis=-1)

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 3
                other_pos.flatten(),  # num_agents-1, 3
                resource_info.flatten(), # num_resources, 3
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
            self.resource_markers = []
            self.labels = []
        
        resource = state.p_pos[self.num_agents:, :2]
        agents = state.p_pos[:self.num_agents, :2]

        # add markers for robots
        if not self.robot_markers:
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(agents[i, 0]),
                    jnp.array(agents[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(self.forage_radius),
                    facecolors='none',
                    edgecolors='green',
                    zorder=-2,
                    linewidth=3
                ) for i in range(self.num_agents)
            ]

            self.labels = [
                self.visualizer.axes.text(
                    agents[i, 0] + 0.2, agents[i, 1] + 0.2, state.het_rep[i, 0],
                    verticalalignment='center', horizontalalignment='center'
                ) for i in range(self.num_agents)
            ]

        # add markers for resources        
        if not self.resource_markers:
            self.resource_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(resource[i, 0]),
                    jnp.array(resource[i, 1]),
                    marker='.',
                    s=self.determine_marker_size(.05),
                    facecolors='black',
                    zorder=-2
                ) for i in range(self.num_resources)
            ]

            self.labels.extend(
                [
                    self.visualizer.axes.text(
                        resource[i, 0] + 0.05, resource[i, 1] + 0.05, state.payload[i],
                        verticalalignment='center', horizontalalignment='center'
                    ) for i in range(self.num_resources)
                ]
            )
        
        
        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(agents[i])
            self.labels[i].set_position(agents[i]+0.2)
        
        # update resource markers
        for i in range(self.num_resources):
            if state.payload[i] < 0:
                self.resource_markers[i].set_sizes([0, 0])
                self.labels[i+self.num_agents].set_text(None)


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
            self.num_agents + self.num_resources,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.3,
        )

        het_rep = self.het_manager.sample(None)

        # set resource levels based on random subsets of the agent levels
        levels = het_rep.flatten()
        level_permutations = jnp.array([levels for _ in range(self.num_resources)])
        resource_mask = jnp.random.permutation(jnp.triu(jnp.ones((self.num_resources, self.num_agents))))
        resource_levels = jnp.sum(level_permutations * resource_mask, axis=-1).flatten()

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            het_rep=het_rep,
            payload=resource_levels # use payload here to track food levels)
        )

        return state

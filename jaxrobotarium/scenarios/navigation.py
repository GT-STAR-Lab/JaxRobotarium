"""
Simple scenario where robots must navigate to goals.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxrobotarium.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class Navigation(RobotariumEnv):
    def __init__(self, num_agents, max_steps=50, **kwargs):
        self.name = 'navigation'
        self.backend = kwargs.get('backend', 'jax')

        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)

        self.pos_shaping = kwargs.get('pos_shaping', -1)
        self.violation_shaping = kwargs.get('violation_shaping', 0)
        self.goal_radius = kwargs.get('goal_radius', 0.1)

        # Observation space
        self.obs_dim = 5
        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }
        
        # Visualization
        self.goal_markers = []

    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # randomly generate initial poses for robots
        poses = generate_initial_conditions(
            2*self.num_agents,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
            key=key
        )
        self.robotarium.poses = poses[:, :self.num_agents]

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
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

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        # check if agents reached goal
        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        d_goal = jnp.linalg.norm(agent_pos - goals, axis=1)
        on_goal = d_goal < self.goal_radius

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full(
                (self.num_agents,),
                jnp.where(jnp.sum(on_goal) < self.num_agents, 0, 1)
            )
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward of distance to goal + violation penalty).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # agent specific shaping reward
        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        d_goal = jnp.linalg.norm(agent_pos - goals, axis=1)
        pos_rew = d_goal * self.pos_shaping

        # global penalty for collisions and boundary violation
        violations = self._get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        return {agent: jnp.where(violation_rew == 0, pos_rew[i], violation_rew) for i, agent in enumerate(self.agents)}

    def get_obs(self, state: State) -> Dict:
        """
        Get observation (pos, vector to goal)

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        to_goal = goals - agent_pos

        return {a: jnp.concatenate([state.p_pos[i], to_goal[i]]) for i, a in enumerate(self.agents)}
    
    #-----------------------------------------
    # Visualization Specific Functions (NOT INTENDED TO BE JITTED)
    #-----------------------------------------

    def render_frame(self, state: State):
        """
        Updates visualizer figure to include goal position markers

        Args:
            state: (State) environment state
        """
        # reset goal markers if at first step
        if state.step == 1:
            self.goal_markers = []

        # add markers for goals        
        goals = state.p_pos[self.num_agents:, :2]
        if not self.goal_markers:
            self.goal_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(goals[i, 0]),
                    jnp.array(goals[i, 1]),
                    marker='.',
                    s=self.determine_marker_size(.05),
                    facecolors='black',
                    zorder=-2
                ) for i in range(self.num_agents)
            ]

    #-----------------------------------------
    # Deployment Specific Functions
    #-----------------------------------------
    def initialize_robotarium_state(self, seed: int = 0):
        """
        Sets initial conditions for robotarium

        Args:
            seed: (int) seed for random functions
        
        Returns:
            (State) initial state
        """

        poses = generate_initial_conditions(
            2*self.num_agents,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5,
        )

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return state

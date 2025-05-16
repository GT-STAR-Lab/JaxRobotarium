# Adding a Scenario
To add a scenario, create a `.py` file with a class definition for your scenario that extends `RobotariumEnv` in this folder. 

At a minimum, your class must implement the following functions: `reset(), step_env(), reward(), get_obs()`. 

To make your scenario deployable on the Robotarium test bed, you must additionally implement `initial_robotarium_state()`, which is intended to randomly or procedureally generate a single starting configuration to be run on the real robots.

Finally, for scenario specific vizualizations (displayed on top of the robot visualization implemented in RobotariumVisualizer), implement `render_frame()`.

# Scenario Library

As a general note for all scenarios, if heterogeneity awareness is configured, robots will additionally observe the configured heterogeneity representation. In our training experiments, the action space for all robots is waypoints that are followed by a unicycle position controller with barrier functions enabled for collision avoidance.

## Arctic Transport

### Description

Adopted from MARBLER, in this scenario, the ice and water robots must cross the terrain consisting of:

- Ice tiles (light blue)  
- Water tiles (dark blue)  
- Ground tiles (white)

to reach the goal zone (green). Ice and water robots move at the same speed across ground tiles, slow over tiles not corresponding to their type, and fast over tiles matching their type.

- Ice and water robots observe:
  - Their own position
  - Positions of other robots
  - The type of tile they are on

- Drone robots:
  - Observe a 3×3 grid centered on their position
  - Move uniformly across all tiles
  - Share observations with ice and water robots to help guide them

The team is rewarded for minimizing the distance of the ice and water robots to the goal zone and penalized for steps where both are not in the goal.

---

## Discovery

### Description
Renamed from **Predator-Capture-Prey** in MARBLER, this scenario involves:

- **N sensing robots**
- **M tagging robots**
- **K landmarks** (black) to be tagged in the environment

Robots observe:

- Their own position
- Positions of other robots
- Initially: placeholder positions for landmarks (i.e., outside the map)

When a landmark enters the range of a sensing robot:
- It becomes sensed (marker turns green)
- Its position is revealed in observations

When it enters the range of a tagging robot:
- It becomes tagged and disappears

Rewards are given for:
- Landmarks sensed
- Landmarks tagged  
Penalties are applied for steps where not all landmarks are tagged.

---

## Foraging

### Description

Inspired by grid-world level-based foraging, in this scenario, N robots with varying levels must collaborate to forage M resources (black markers) within the environment. Labels indicate the levels of robots and resources.

Robots observe:

- Their own position
- Positions of other robots
- Locations of resources
- Their individual foraging levels

To forage a resource:
- It must be within the foraging radius of M robots
- The sum of the robot levels must be greater than or equal to the level of the resource

When a resource is foraged, it disappears from the environment. The team is rewarded based on the level of the resources collected. This scenario is scalable to arbitrary numbers of robots and resources.

---

## Material Transport

### Description

Adopted from MARBLER, this scenario involves N robots with varying speeds and carrying capacities collaborating to unload two loading zones (green circle and green rectangle) into a drop-off zone (purple rectangle).

- Both zones are initialized with material sampled from a Gaussian distribution
- Robots can load material by entering a loading zone
- Robots can carry up to their own capacity
- Robots must unload into the drop-off zone before reloading

Robots observe:

- Their own position
- Positions of other robots
- Remaining material in both zones

The team is rewarded for material loaded and dropped off, and penalized when material remains in the loading zones. The scenario scales with the number of robots.

---

## Navigation (MAPF)

### Description

A classic Multi-Agent Path Finding (MAPF) scenario where N robots must each navigate to individually assigned goals (black markers).

Robots observe:

- Their own position
- Vector to their goal

The team is rewarded for minimizing the Euclidean distance between each robot and its respective goal. The scenario is scalable to arbitrary numbers of robots.

---

## Predator Prey

### Description

Based on a variant of the classic predator-prey scenario, N tagging robots must collaborate to tag a more agile prey (green marker).

- The prey is controlled by a heuristic that maximizes its distance from the closest tagging robot at each timestep
- Robots observe:
  - Their own position
  - Positions of other robots
  - Position of the prey

Robots tag the prey by getting it within their tagging radius. A successful tag briefly turns the prey red. The team is rewarded for each successful tag. This scenario scales to any number of robots.

---

## Continuous-RWARE

### Description

Inspired by grid-world RWARE, robots collaborate to deliver requested shelves to a drop-off zone (purple). Shelf requests are shown in the interface and update after each successful delivery.

- Shelves begin in gray staging zones
- Robots can:
  - Pick up shelves from staging zones
  - Return them to unoccupied zones
  - Drop them off at the purple zone if requested

Robots observe:

- Their own position
- Positions of other robots
- Staging zones and shelf locations
- Current shelf requests

Robots can move freely without a shelf but must avoid blocked paths when carrying one. Sparse rewards are given for successful drop-offs. The scenario scales with the number of robots and shelves.

---

## Warehouse

### Description

Adopted from MARBLER, this scenario involves N red robots and M green robots picking up packages from a color-matched zone on the right and delivering them to the corresponding zone on the left.

Robots observe:

- Their own position
- Positions of other robots

The team is rewarded for both loading and delivering packages. The scenario supports scalability to any number of robots.


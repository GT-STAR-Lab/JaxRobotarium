# Deploying to the Robotarium

## Required Files
1. `main.py`: script that executes scenario, called by robotarium
2. `robotarium_env.py`: identical to training `robotarium_env`, but interfaces with robotarium python simulator instead of the jax backend
3. `<actor>.py`: actor file with pytorch implementation of your flax model (see [Notable Assumptions](#notable-assumptions) for additional details)
4. `<actor_weights>.safetensors`: model weights
5. `<scenario>.py`: backend agnostic scenario file, to achieve this add the following to dynamically import jax or python backend (will be copied from `marbler` folder when calling `deploy.py`)
    ```
    # wrap import statement in try-except block to allow for correct import during deployment
    try:
        from marbler.robotarium_env import *
    except Exception as e:
        from robotarium_env import *
    ```
6. `constants.py`: global robotarium constants (will be copied from `marbler` folder when calling `deploy.py`)

## Deployment Steps
1. Set `config.yaml`
    ```
    "seed": # random seed

    # ENV
    "scenario": # scenario class name (should match filename when lowercase)
    "num_agents": # number of agents
    "max_steps": # maximum number of steps to run scenario for for
    "action_type": # "Discrete" or "Continuous"
    "number_of_robots": # number of robots TODO: this is pretty redunant
    "controller": # controller to use, null if unused
    "barrier_fn": # barrier function to use, null if unused
    "update_frequency": # number of steps in the robotarium per simulator step
    "backend": "python" # needs to be not "jax"
    "save_gif": # whether to save a gif of the rollout
    "preprocess_obs": # whether to add agent id (carried over jaxmarl parameter)

    # MODEL CONFIG
    "input_dim": # observation dimension (ignoring agent id)
    "hidden_dim": # hidden dimension
    "output_dim": # action dimension

    # DEPLOYMENT CONFIG
    "model_weights": # path to model weights
    "model_file": # path to actor file
    "model_class": # name of pytorch actor class to import
    ```
2. Run `python deploy.py --name <experiment_name> --config <config.yaml>` in this directory to generate the folder containing all files necessary for deployment at `robotarium_submissions/<experiment_name>` using configuration in `config.yaml`
3. Upload files to [Robotarium](https://www.robotarium.gatech.edu/)
    * Create new experiment
    * Set number of robots to match your config file
    * Upload all files in your generated experiment folder to "Experiment Files" section
    * Select `main.py` as "Main File"
    * Submit!

## Notable Assumptions
1. You have created an actor file that matches your flax model
    * Layer names match
        ```
        # in flax
        nn.Dense

        # in pytorch (0 for 1st dense layer used in flax, 1 for 2nd dense layer, etc.)
        self.Dense_0 = nn.Linear
        ```
    * inputs, outputs, and dimensions must match at all layers
2. Your defined actor has defined conversions from flax to pytorch parameter conventions
    * currently have implemented conversions for `Dense` and `GRUCell`
    * see `flax_to_torch` in `deploy.py` for implementation details and to implement additional layer types

For an example of how to test a model conversion, see the provided example in `deploy/test`.

## Miscellaneous
* To verify your experiment will run, you can run the `main.py` file within your generated experiment folder and confirm no exceptions are thrown, etc. Additionally, you can set `"save_gif"` to true to see what your executed experiment will look like.
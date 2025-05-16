import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_dataframe(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")

def load_dataframe(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    else:
        print(f"File {filename} not found.")
        return None

def fetch_wandb_data(run_paths, metrics):
    api = wandb.Api(timeout=60)
    all_run_data = []

    for path, metric in zip(run_paths, metrics):
        try:
            run = api.run(path)
            config = run.config

            keys = [metric, '_runtime']
            print(f"Fetching: {run.name} with metrics {keys} from {path}")
            history = run.scan_history()

            run_data = pd.DataFrame(history)
            run_data['run_path'] = path
            run_data['run_name'] = run.name

            if 'jax' in path:
                num_envs = config.get("NUM_ENVS", None)
                if num_envs:
                    run_data['run_type'] = f"JaxRobotarium ({num_envs})"
                else:
                    run_data['run_type'] = "JaxRobotarium (num_envs=?)"
            else:
                run_data['run_type'] = 'MARBLER'

            # Normalize metric naming
            run_data['return'] = run_data[metric[0]] * (config["NUM_STEPS"]-1) if 'jax' in path else run_data[metric[0]] / 4 # normalization
            run_data['_runtime'] = run_data['_runtime'] - run_data['_runtime'].min()
            run_data['timestep'] = run_data[metric[1]]

            all_run_data.append(run_data)
        except Exception as e:
            print(f"Failed to fetch run at {path}: {e}")

    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb(name, run_paths, metrics):
    df = fetch_wandb_data(run_paths, metrics)
    filename = f"{name}.pkl" 
    save_dataframe(df, filename)
    return df

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def plot_metric_over_wall_time(df, title, name, metric, swap_axes=False, legend=True, smooth_sigma=10):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 24})
    palette = {
        "JaxRobotarium (8)": "#FF6365", "JaxRobotarium (1)": "#B4A7D6", "MARBLER": sns.color_palette()[0],
    }

    # Determine shared runtime upper bound (minimum of all run max runtimes)
    max_runtimes = df.groupby('run_path')['_runtime'].max()
    global_max_time = max_runtimes.min() - 50

    # Interpolation grid
    common_times = np.linspace(0, global_max_time, 1000)

    for i, (label, group) in enumerate(df.groupby('run_type')):
        print(label)
        runs = []

        for run_id, run_data in group.groupby('run_path'):
            run_data_sorted = run_data.sort_values('_runtime')
            x = run_data_sorted['_runtime'].values
            y = run_data_sorted[metric].values

            # Remove NaNs
            valid = (~np.isnan(x)) & (~np.isnan(y)) & (x <= global_max_time)
            x = x[valid]
            y = y[valid] * 4 if (label == "MARBLER" and "warehouse" in name.lower() and "return" in metric) else y[valid]

            try:
                interp_func = interp1d(
                    x, y,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                interpolated = interp_func(common_times)
                interpolated = interpolated[common_times <= global_max_time]
                runs.append(interpolated)
            except Exception as e:
                print(f"Skipping run {run_id} due to interpolation error: {e}")

        if not runs:
            continue

        runs_array = np.stack(runs)
        mean_vals = np.mean(runs_array, axis=0)
        std_vals = np.std(runs_array, axis=0)

        # Apply Gaussian smoothing
        mean_vals = gaussian_filter1d(mean_vals, sigma=smooth_sigma)
        std_vals = gaussian_filter1d(std_vals, sigma=smooth_sigma)

        color = palette[label]
        if swap_axes:
            plt.plot(mean_vals, common_times, label=label, color=color, linewidth=2)
            plt.fill_betweenx(common_times, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)
        else:
            plt.plot(common_times, mean_vals, label=label, color=color, linewidth=2)
            plt.fill_between(common_times, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2)

    if swap_axes:
        plt.ylabel("Wall Time (s)")
        plt.xlabel(metric.replace("_", " ").title())
    else:
        plt.xlabel("Wall Time (s)")
        plt.ylabel(metric.replace("_", " ").title())

    plt.title(title)
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,0,2]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    # Example usage

    # DISCOVERY
    # run_paths = [
    #     "star-lab-gt/jax-marbler/812z3rcf",
    #     "star-lab-gt/jax-marbler/bpacmajk",
    #     "star-lab-gt/jax-marbler/aqxp3fdo",
    #     "star-lab-gt/jax-marbler/gpribpfg",
    #     "star-lab-gt/jax-marbler/yuoelbw6",
    #     "star-lab-gt/jax-marbler/9sl74oug",
    #     "star-lab-gt/CASH-MARBLER/yeg52ads",
    #     "star-lab-gt/CASH-MARBLER/b83amt8u",
    #     "star-lab-gt/CASH-MARBLER/cqds68ae"
    # ]
    # title = 'Discovery'
    # name = "qmix-discovery"
    # metrics = [
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    # ]  # specify the metric key(s)
    # df = load_dataframe(f"{name}.pkl")
    # if df is None:
    #     df = get_from_wandb(name, run_paths, metrics)

    # plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    # plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", legend=False)

    # # MT
    # run_paths = [
    #     "star-lab-gt/jax-marbler/mryj8x39",
    #     "star-lab-gt/jax-marbler/nl54llwl",
    #     "star-lab-gt/jax-marbler/pnzpnzf5",
    #     "star-lab-gt/jax-marbler/nig5ue9o",
    #     "star-lab-gt/jax-marbler/dcogye3e",
    #     "star-lab-gt/jax-marbler/a8yda4bx",
    #     "star-lab-gt/CASH-MARBLER/jwlw2n5h",
    #     "star-lab-gt/CASH-MARBLER/dehqnogb",
    #     "star-lab-gt/CASH-MARBLER/8dztwrej"
    # ]
    # metrics = [
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    # ]  # specify the metric key(s) # specify the metric key(s)
    # title = 'Material Transport'
    # name = "qmix-mt"
    # df = load_dataframe(f"{name}.pkl")
    # if df is None:
    #     df = get_from_wandb(name, run_paths, metrics)

    # plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    # plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", legend=False)

    # WAREHOUSE
    run_paths = [
        "star-lab-gt/jax-marbler/hk08p909",
        "star-lab-gt/jax-marbler/hh63wnvl",
        "star-lab-gt/jax-marbler/m1zbhtuh",
        "star-lab-gt/jax-marbler/i5bpenri",
        "star-lab-gt/jax-marbler/4frlovid",
        "star-lab-gt/jax-marbler/vb48bn5c",
        "star-lab-gt/CASH-MARBLER/kfzdi5ka",
        "star-lab-gt/CASH-MARBLER/lqop1sb0",
        "star-lab-gt/CASH-MARBLER/cl1tc7ra"
    ]
    metrics = [
        ["returned_episode_returns", "env_step"],
        ["returned_episode_returns", "env_step"],
        ["returned_episode_returns", "env_step"],
        ["returned_episode_returns", "env_step"],
        ["returned_episode_returns", "env_step"],
        ["returned_episode_returns", "env_step"],
        ["return_mean", "_step"],
        ["return_mean", "_step"],
        ["return_mean", "_step"],
    ]  # specify the metric key(s) # specify the metric key(s)
    title = 'Warehouse'
    name = "qmix-warehouse"
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    # plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    # plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", legend=False)

    # # ARCTIC TRANSPORT
    # run_paths = [
    #     "star-lab-gt/jax-marbler/0q341a32p",
    #     "star-lab-gt/jax-marbler/nlws6c72",
    #     "star-lab-gt/jax-marbler/ji08lrmm",
    #     "star-lab-gt/jax-marbler/7mfy0mr9",
    #     "star-lab-gt/jax-marbler/lc3z0w5y",
    #     "star-lab-gt/jax-marbler/fo0qmtzw"
    #     "star-lab-gt/CASH-MARBLER/j1lopo5u",
    #     "star-lab-gt/CASH-MARBLER/jfsr6zsa",
    #     "star-lab-gt/CASH-MARBLER/1d96evpi"
    # ]
    # metrics = [
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["returned_episode_returns", "env_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    #     ["return_mean", "_step"],
    # ]  # specify the metric key(s)
    # title = 'Arctic Transport'
    # name = "qmix-arctic-transport"
    # df = load_dataframe(f"{name}.pkl")
    # if df is None:
    #     df = get_from_wandb(name, run_paths, metrics)

    # plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=True)
    # plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", legend=False)
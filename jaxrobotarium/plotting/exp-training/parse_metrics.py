import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

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

def fetch_wandb_data(project_name, tags, metric_prefix="test_returned_episode_returns"):
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, {"tags": {"$in": tags}})
    print(f"found {len(runs)} runs with tags {tags}")

    if len(runs) == 0:
        return pd.DataFrame()

    # Step 1: Discover all matching metrics from the first run
    sample_history = runs[0].history(samples=10000)
    matching_metrics = [key for key in sample_history.columns if re.match(r"rng.*/" + re.escape(metric_prefix), key)]
    if "env_step" not in matching_metrics:
        matching_metrics.append("env_step")

    print(f"Found {len(matching_metrics)} matching metrics: {matching_metrics}")

    all_run_data = []
    for run in runs:
        try:
            # Step 2: Pull only matching metrics
            history = run.scan_history()
            run_data = pd.DataFrame(history)

            run_data['run_id'] = run.id
            run_data['run_name'] = run.name
            run_data['timestep'] = run_data['env_step']
            run_data['tags'] = ', '.join(run.tags)
            run_data['env_name'] = run.config.get("ENV_NAME")

            all_run_data.append(run_data)
        except Exception as e:
            print(f"Failed to fetch run {run.name} ({run.id}): {e}")

    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb(tags, metric_prefix="test_returned_episode_returns"):
    project_name = "jax-marbler"

    df = fetch_wandb_data(project_name, tags, metric_prefix=metric_prefix)
    filename = f"{tags[0]}.pkl" 
    save_dataframe(df, filename)

    print("saved")
    print(df.head())

def smooth_and_downsample(df, y_column, mean_window=50, std_window=50, downsample_factor=10):
    """
    Creates a new dataframe with smoothed and downsampled data, with separate
    smoothing controls for mean and standard deviation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_column : str
        Column to analyze
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    smoothed_data = []
    df_copy = df.copy()

    for baseline in df_copy['run_id'].unique():
        baseline_data = df_copy[df_copy['run_id'] == baseline].copy()
        baseline_data = baseline_data.sort_values('timestep')

        # grouped = baseline_data.groupby('timestep')[y_column] # .agg(['mean', 'std']).reset_index()
        max_timesteps = baseline_data['timestep'].max()
        max_timesteps_rows = baseline_data[baseline_data['timestep'] == max_timesteps]
        cols = [y_column, 'run_id', 'run_name']
        print("ALL SEEDS metrics:")
        print(max_timesteps_rows[cols])
        print()

        # Group by timestep to calculate mean and std
        grouped = baseline_data.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()

        # Smooth mean and std separately
        grouped['smooth_mean'] = grouped['mean'].rolling(
            window=mean_window, min_periods=1, center=True).mean()
        grouped['smooth_std'] = grouped['std'].rolling(
            window=std_window, min_periods=1, center=True).mean()

        # Downsample
        grouped = grouped.iloc[::downsample_factor]

        # Create dataframe with smoothed mean and smoothed std
        smoothed_df = pd.DataFrame({
            'timestep': grouped['timestep'],
            f'{y_column}': grouped['smooth_mean'],
            f'{y_column}_std': grouped['smooth_std'],
            'run_id': baseline
        })

        smoothed_data.append(smoothed_df)

    return pd.concat(smoothed_data)


def plot_comparison_by_env(df_qmix, df_pqn, df_mappo, df_ippo, y_column_prefix='test_returned_episode_returns',
                            mean_window=1, std_window=1, downsample_factor=1):
    df_qmix = df_qmix.copy()
    df_pqn = df_pqn.copy()
    df_mappo = df_mappo.copy()
    df_ippo = df_ippo.copy()
    df_qmix['algorithm'] = 'QMIX'
    df_pqn['algorithm'] = 'PQN'
    df_mappo['algorithm'] = 'MAPPO'
    df_ippo['algorithm'] = 'IPPO'
    colors = sns.color_palette()
    color_legend = {
        "QMIX": colors[0], "PQN": colors[1], "MAPPO": colors[2], "IPPO": colors[3]
    }

    combined_df = pd.concat([df_qmix, df_pqn, df_mappo, df_ippo], ignore_index=True)

    # Identify all return columns matching rng.../test_returned_episode_returns
    return_cols = [col for col in combined_df.columns if re.match(r"rng.*/" + re.escape(y_column_prefix), col)]

    # Melt all rng-specific return columns into one long-form column
    df_long = combined_df.melt(
        id_vars=['timestep', 'run_id', 'run_name', 'env_name', 'tags', 'algorithm'],
        value_vars=return_cols,
        var_name='rng_seed',
        value_name='return'
    )

    env_names = df_long['env_name'].unique()
    for env in env_names:
        env_df = df_long[df_long['env_name'] == env]

        smoothed_data = []

        for (run_id, algo), group in env_df.groupby(['run_id', 'algorithm']):
            group = group.sort_values('timestep')

            # First, aggregate across RNG seeds: mean and std at each timestep
            grouped = group.groupby('timestep')['return'].agg(['mean', 'std']).reset_index()

            # Smooth the aggregated mean and std
            grouped['smooth_mean'] = grouped['mean'].rolling(
                window=mean_window , min_periods=1, center=True).mean()
            grouped['smooth_std'] = grouped['std'].rolling(
                window=std_window, min_periods=1, center=True).mean()

            # Downsample
            factor = downsample_factor
            grouped = grouped.iloc[::factor]

            grouped['algorithm'] = algo
            grouped['env_name'] = env

            smoothed_data.append(grouped)

        final_df = pd.concat(smoothed_data, ignore_index=True)

        plt.figure(figsize=(8, 6))
        plt.rc('font', size=18)

        for algo in final_df['algorithm'].unique():
            algo_df = final_df[final_df['algorithm'] == algo]
            plt.plot(algo_df['timestep'], algo_df['smooth_mean'], label=algo, color=color_legend[algo])
            plt.fill_between(
                algo_df['timestep'],
                algo_df['smooth_mean'] - algo_df['smooth_std'],
                algo_df['smooth_mean'] + algo_df['smooth_std'],
                alpha=0.3,
                color=color_legend[algo]
            )

        plt.title(f"{' '.join(env.split('_')).title()}")
        plt.xlabel("Timestep")
        plt.ylabel("Return")
        plt.legend(title=None)
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{env}.png')
        # plt.show()  # Uncomment if viewing interactively


if __name__ == "__main__":
    # tags = ['final-qmix']
    # get_from_wandb(tags)

    # tags = ['final-pqn']
    # get_from_wandb(tags)

    # tags = ['final-mappo']
    # get_from_wandb(tags)

    tags = ['final-ippo']
    get_from_wandb(tags)

    df_qmix = load_dataframe('final-qmix.pkl')
    df_pqn = load_dataframe('final-pqn.pkl')
    df_mappo = load_dataframe('final-mappo.pkl')
    df_ippo = load_dataframe('final-ippo.pkl')

    plot_comparison_by_env(df_qmix, df_pqn, df_mappo, df_ippo)
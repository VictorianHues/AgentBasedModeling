"""Script to run cluster analysis for varying parameters."""

import os

import numpy as np

from abm_project.cluster_analysis import (
    cluster_time_series,
)
from abm_project.plotting_cluster_analysis import (
    plot_eq_env_against_memory_rationality,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def eq_env_against_memory_rationality(
    models=None, memory_range=None, rat_range=None, savedir=None
):
    """Calculate the equilibrium environment state for varying parameters.

    Args:
        models: List of VectorisedModel instances with varying
                memory counts and rationalities.
        memory_range: List of memory counts to test.
        rat_range: List of rationality values to test.
        savedir: Directory to save the npz file. If None,
                    saves in current directory.
    """
    # savedir = Path(savedir).mkdir(parents=True, exist_ok=True)
    # savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    eq_env_state = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model:
                history = model.environment[: model.time + 1]
                history = history.reshape((-1, model.height, model.width))
                lattice = history[history.shape[0] - 1]
                eq_env_state[i, j] = np.mean(lattice)
    print(f"eq_env_state shape: {eq_env_state.shape}")
    print(f"eq_env_state: {eq_env_state}")

    # Save equilibrium state to a npz file
    os.makedirs(savedir, exist_ok=True)
    filepath = savedir + "/eq_env_state.npz"
    np.savez(
        filepath,
        eq_env_state=eq_env_state,
        memory_range=memory_range,
        rat_range=rat_range,
    )

    return None


def nclusters_against_memory_rationality(
    models=None, option="environment", memory_range=None, rat_range=None, savedir=None
):
    """Calculate the number of clusters for varying memory and rationality.

    Args:
        models: List of VectorisedModel instances with varying
                memory counts and rationalities.
        option: Option indicating the type of analysis (e.g., "environment").
        memory_range: List of memory counts to test.
        rat_range: List of rationality values to test.
        savedir: Directory to save the npz file. If None,
                    saves in current directory.
    """
    # savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    n_clusters = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model:
                _, nc, _ = cluster_time_series(model=model, option=option)
                n_clusters[i, j] = np.mean(nc)  # Average number of clusters over time
    print(f"n_clusters shape: {n_clusters.shape}")

    # Save equilibrium state to a npz file
    os.makedirs(savedir, exist_ok=True)
    filepath = savedir + "/n_clusters.npz"
    np.savez(
        filepath,
        n_clusters=n_clusters,
        memory_range=memory_range,
        rat_range=rat_range,
    )

    return None


if __name__ == "__main__":
    # Model parameters
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000

    memory_count_range = [a for a in np.arange(2, 52, 2)]
    rationality_range = np.linspace(0, 1, 30, endpoint=True)
    print(rationality_range)
    # Uncomment the following lines to use specific memory_count and rationality:
    # memory_count = 20
    # rationality = 0.9
    gamma_s = 0.05

    # Environment update parameters
    recovery = 1
    pollution = 1
    gamma = 0.01

    env_update_fn = piecewise_exponential_update(
        recovery=recovery, pollution=pollution, gamma=gamma
    )

    rng = None
    neighb_prediction_option = "linear"  # or "logistic"
    severity_benefit_option = "adaptive"  # or None

    models = []

    # Create models for all combinations of memory_count and rationality
    for memory_count in memory_count_range:
        for rationality in rationality_range:
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=rationality,
                gamma_s=gamma_s,
                neighb_prediction_option=neighb_prediction_option,
                severity_benefit_option=severity_benefit_option,
                max_storage=num_steps,
            )
            models.append(model)

    # Uncomment the following lines to use specific memory_count and rationality:
    # print(f"Using memory_count={memory_count} and rationality={rationality}")
    # # Create a single model with specified memory_count and rationality
    # model = VectorisedModel(
    #     num_agents=num_agents,
    #     width=width,
    #     height=height,
    #     memory_count=memory_count,
    #     rng=rng,
    #     env_update_fn=env_update_fn,
    #     rationality=rationality,
    #     gamma_s=gamma_s,
    #     neighb_prediction_option=neighb_prediction_option,
    #     severity_benefit_option=severity_benefit_option,
    #     max_storage=num_steps,
    # )
    # models.append(model)

    # models
    print(f"Generated {len(models)} models.")

    # Run the code below only if more than 1 model is generated
    filepath_eq_env = "data/eq_env_vs_memory_rationality"
    eq_env_against_memory_rationality(
        models=models,
        memory_range=memory_count_range,
        rat_range=rationality_range,
        savedir=filepath_eq_env,
    )
    print("Plotting equilibrium environment state...")
    plot_eq_env_against_memory_rationality(
        filepath=filepath_eq_env + "/eq_env_state.npz",
        savedir="plots/eq_env_vs_memory_rationality/",
    )

    # filepath_nclusters = "data/n_clusters_vs_memory_rationality"
    # nclusters_against_memory_rationality(
    #     models=models,
    #     option="environment",
    #     memory_range=[m.memory_count for m in models],
    #     rat_range=[m.rationality for m in models],
    #     savedir=filepath_nclusters,
    # )
    # print("Plotting number of clusters against memory and rationality...")
    # plot_nclusters_against_memory_rationality(
    #     filepath=filepath_nclusters + "/n_clusters.npz",
    #     savedir="plots/n_clusters_vs_memory_rationality/",
    # )

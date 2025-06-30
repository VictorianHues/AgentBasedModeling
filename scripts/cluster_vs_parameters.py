"""Script to investigate cluster formation across varying parameters."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.cluster_analysis import cluster_time_series
from abm_project.plotting_cluster_analysis import (
    plot_eqenv_across_memory_rationality,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def calc_eqenv_across_memory_rationality(
    models, memory_range, rationality_range, save_as_name=None, savedir=None
):
    """Calculate the equilibrium env for different memory and rationality levels.

    Args:
        models (list): List of VectorisedModel instances.
        memory_range (list): List of memory counts to consider.
        rationality_range (list): List of rationality levels to consider.
        savedir (Path, optional): Directory to save the results.
                                    Defaults to current directory.

    Returns:
        None: Saves the equilibrium environment status to a npz file.
    """
    savedir = savedir or Path(".")

    # Initialize the equilibrium environment status array
    eq_env_status = np.zeros((len(memory_range), len(rationality_range)))

    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rationality_range):
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model is None:
                print(
                    f"No model found for memory {mem} and rationality {rat}. Skipping."
                )
                continue
            # Compute the equilibrium environment status
            history = model.environment[: model.time + 1]
            history = history.reshape((-1, model.height, model.width))
            last_timestep = history.shape[0] - 1
            eq_env_status[i, j] = np.mean(history[last_timestep])

    # Save the equilibrium environment status to a npz file
    filename = save_as_name + ".npz"
    np.savez(
        savedir / filename,
        eq_env_status=eq_env_status,
        memory_range=memory_range,
        rationality_range=rationality_range,
    )


def calc_cluster_across_memory_rationality(
    models, savedir=None, memory_range=None, rationality_range=None, save_as_name=None
):
    """Calculate the number of clusters across different parameters."""
    savedir = savedir or Path(".")

    # Initialize the number of clusters array
    ncluster = np.zeros((len(memory_range), len(rationality_range)))

    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rationality_range):
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model is None:
                print(
                    f"No model found for memory {mem} and rationality {rat}. Skipping."
                )
                continue
            # Compute the number of clusters at each time step
            _, nc, _ = cluster_time_series(model=model, option="environment")
            ncluster[i, j] = np.mean(nc)

    # Save the number of clusters to a npz file
    filename = save_as_name + ".npz"
    np.savez(
        savedir / filename,
        ncluster=ncluster,
        memory_range=memory_range,
        rationality_range=rationality_range,
    )


if __name__ == "__main__":
    option = "environment"

    # Set model parameters
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000

    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = None

    memory_count_range = [a for a in np.arange(10, 151, 10)]
    rationality_range = np.linspace(0.01, 2.0, 15)

    gamma_s = 0.004

    models = []

    for memory_count in memory_count_range:
        for rationality in tqdm(
            rationality_range, desc=f"Memory Count {memory_count}, Rationality: "
        ):
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=rationality,
                gamma_s=gamma_s,
                neighb_prediction_option="linear",
                severity_benefit_option="adaptive",
                max_storage=num_steps,
            )
            model.run(num_steps)
            models.append(model)

    # start = time.time()
    # model = VectorisedModel(
    #     num_agents=num_agents,
    #     width=width,
    #     height=height,
    #     memory_count=memory_count,
    #     rng=rng,
    #     env_update_fn=env_update_fn,
    #     rationality=rationality,
    #     gamma_s=gamma_s,
    #     neighb_prediction_option="linear",
    #     severity_benefit_option="adaptive",
    #     max_storage=num_steps,
    # )
    # model.run(num_steps)
    # end = time.time()

    # print(f"Simulation completed in {end - start:.2f} seconds.")

    # Phase plot of equilibrium environment status
    # across memory and rationality levels:
    data_eqenv_filename = "eq_env_state_memory_rationality"

    calc_eqenv_across_memory_rationality(
        models=models,
        memory_range=memory_count_range,
        rationality_range=rationality_range,
        save_as_name=data_eqenv_filename,
        savedir=Path("data"),
    )

    plot_eqenv_across_memory_rationality(
        data_file=Path("data/" + data_eqenv_filename + ".npz"), savedir=Path("plots")
    )

    # Phase plot of number of clusters
    # across memory and rationality levels:
    # data_ncluster_filename = "n_clusters_memory_rationality"

    # calc_cluster_across_memory_rationality(
    #     models=models,
    #     memory_range=memory_count_range,
    #     rationality_range=rationality_range,
    #     save_as_name=data_ncluster_filename,
    #     savedir=Path("data")
    # )

    # plot_cluster_across_memory_rationality(
    #     data_file=Path("data/" + data_ncluster_filename + ".npz"),
    #     savedir=Path("plots")
    # )

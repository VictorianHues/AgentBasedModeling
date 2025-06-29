"""Batch script to run the vectorised model and generate animations of the results."""

import concurrent.futures

import numpy as np

from abm_project.batch_run_tools import (
    clustering_score_over_time,
)
from abm_project.plotting import (
    get_data_directory,
    plot_list_over_time,
    plot_mean_and_variability_array,
)
from abm_project.vectorised_model import VectorisedModel


def run_single_simulation(args):
    print("Running single simulation")
    steps, kwargs = args
    model = VectorisedModel(**kwargs)
    model.run(steps)
    env_mean = np.mean(model.environment, axis=1)
    action_mean = np.mean(model.action, axis=1)
    cluster_score = clustering_score_over_time(
        model,
        attribute="action",
        width=kwargs["width"],
        height=kwargs["height"],
        radius=1,
    )
    del model
    return env_mean, action_mean, cluster_score


def run_single_parameter_set():
    num_runs = 500  # number of runs for the batch simulation
    steps = 10000  # number of simulation steps
    width = 30  # width of the grid
    height = 30  # height of the grid
    num_agents = width * height

    kwargs = {
        "num_agents": num_agents,
        "width": width,
        "height": height,
        "memory_count": 10,
        "rng": None,
        "rationality": 1.0,
        "max_storage": steps,
        "moore": True,
        "simmer_time": 1,
        "neighb_prediction_option": None,  # "linear",
        "severity_benefit_option": "adaptive",
        "radius_option": "all",
        "prop_pessimistic": 1.0,
        "pessimism_level": 1.0,
        "b_1": np.full(num_agents, 1.0, dtype=np.float64),
        "b_2": np.full(num_agents, 1.0, dtype=np.float64),
        "gamma_s": 0.01,
    }

    env_means = np.zeros((num_runs, steps + 1))
    env_vars = np.zeros((steps + 1,))
    action_means = np.zeros((num_runs, steps + 1))
    action_vars = np.zeros((steps + 1,))
    action_cluster_scores = np.zeros(
        (
            num_runs,
            steps + 1,
        )
    )

    run_args = [(steps, kwargs) for _ in range(num_runs)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_simulation, run_args))

    for run, (env_mean, action_mean, cluster_score) in enumerate(results):
        env_means[run, :] = env_mean
        action_means[run, :] = action_mean
        action_cluster_scores[run, :] = cluster_score

    data_file_path = get_data_directory("batch_run_results.npz")
    np.savez(
        data_file_path,
        env_means=env_means,
        action_means=action_means,
        cluster_scores=action_cluster_scores,
    )

    plot_mean_and_variability_array(
        env_means, title="Environment Over Time", file_name="env_over_time.png"
    )
    plot_mean_and_variability_array(
        action_means, title="Actions Over Time", file_name="action_over_time.png"
    )
    plot_mean_and_variability_array(
        action_cluster_scores,
        title="Spatial Clustering Over Time",
        file_name="action_cluster_scores_over_time.png",
    )

    env_vars = np.var(env_means, axis=0)
    action_vars = np.var(action_means, axis=0)
    action_cluster_score_vars = np.var(action_cluster_scores, axis=0)

    plot_list_over_time(
        env_vars,
        title="Variance of Environment Status Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="env_variance_over_time.png",
    )
    plot_list_over_time(
        action_vars,
        title="Variance of Actions Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="action_variance_over_time.png",
    )
    plot_list_over_time(
        action_cluster_score_vars,
        title="Variance of Action Cluster Scores Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="action_cluster_score_variance_over_time.png",
    )


if __name__ == "__main__":
    run_single_parameter_set()

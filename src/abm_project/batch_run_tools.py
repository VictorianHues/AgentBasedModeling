"""Batch run tools for ABM project."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from abm_project.oop_model import BaseModel


def run_batch(
    num_runs: int, model_class, steps: int = 100, **kwargs
) -> list[BaseModel]:
    """Run a batch of agent-based model simulations.

    Args:
        num_runs (int): Number of model runs to execute.
        model_class (type): The class of the model to instantiate.
        steps (int): Number of simulation steps to run for each model.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        list[BaseModel]: List of model instances after running the simulation.
    """
    models = []
    for _ in range(num_runs):
        rng = np.random.default_rng()
        model = model_class(rng=rng, **kwargs)
        model.run(steps)
        models.append(model)
    return models


def extract_mean_and_variance(models, attribute: str):
    """Extract the mean and variance of a specified attribute.

    Args:
        models (list): List of BaseModel instances.
        attribute (str): The name of the attribute history
            to extract (e.g., "agent_action_history").

    Returns:
        tuple: (means, variances), where each is a list of values per time step.
    """
    # Determine the number of time steps in the simulation
    num_steps = len(models[0].__getattribute__(attribute))

    means = []
    variances = []

    # Iterate through each time step
    for t in range(num_steps):
        # Get the attribute grid at time t from each model
        grids = np.array([model.__getattribute__(attribute)[t] for model in models])

        # Compute the mean and variance of values across the grids
        step_means = np.array([np.mean(grid) for grid in grids])
        step_vars = np.array([np.var(grid) for grid in grids])

        # Record the average mean and variance across all model runs
        means.append(np.mean(step_means))
        variances.append(np.mean(step_vars))

    return means, variances


def average_metric_over_time(models, attr, inner_mean=False):
    """Calculate the average of a given attribute over time across all models.

    Args:
        models (list[BaseModel]): List of BaseModel instances.
        attr (str): Attribute name to average (e.g., 'agent_action_history').
        inner_mean (bool): If True, take mean over
            inner array before averaging across models.

    Returns:
        list: Average value at each time step.
    """
    history_length = len(getattr(models[0], attr))
    if inner_mean:
        # For each time step, compute the mean of the inner array for each model,
        # then average these means across all models.
        result = []
        for t in range(history_length):
            inner_means = []
            for m in models:
                # Get the attribute at time t for model m (should be an array-like)
                values = getattr(m, attr)[t]
                mean_value = np.mean(values)
                inner_means.append(mean_value)
                # Average the means across all models for this time step
                avg_mean = np.mean(inner_means)
                result.append(avg_mean)
        return result
    else:
        return [
            np.mean([getattr(m, attr)[t] for m in models])
            for t in range(history_length)
        ]


def attribute_variance_over_time(models, attr):
    """Calculate the mean variance of a given attribute over time across all models.

    Args:
        models (list[BaseModel]): List of BaseModel instances.
        attr (str): Attribute name to compute variance over
            (e.g., 'agent_peer_pressure_coeff_history').

    Returns:
        list: Mean variance at each time step.
    """
    history_length = len(getattr(models[0], attr))
    return [
        np.mean([np.var(getattr(m, attr)[t]) for m in models])
        for t in range(history_length)
    ]


def spatial_clustering_over_time(models, radius=1):
    """Calculate the spatial clustering score over time for a batch of models.

    Args:
        models (list[BaseModel]): List of BaseModel instances.
        radius (int): Radius for local averaging in the spatial clustering score.

    Returns:
        list: Spatial clustering scores for each time step.
    """
    return [
        np.mean(
            [
                spatial_clustering_score(m.agent_action_history[t], radius)
                for m in models
            ]
        )
        for t in range(len(models[0].agent_action_history))
    ]


def spatial_clustering_score(grid, radius=1):
    """Calculate the spatial clustering score of a grid.

    The score is based on the local average of binary values in the grid.

    Args:
        grid (np.ndarray): 2D array representing the grid.
        radius (int): Radius for local averaging.

    Returns:
        float: Spatial clustering score.
    """
    binary = (grid + 1) / 2  # map [-1,1] to [0,1]
    local_avg = uniform_filter(binary, size=2 * radius + 1, mode="wrap")
    return np.mean(np.abs(binary - local_avg))


def plot_spatial_clustering_score_heatmap(
    clustering_scores, title="Spatial Clustering Score Heatmap"
):
    """Plot a heatmap of spatial clustering scores.

    Args:
        clustering_scores (list): List of spatial clustering scores for each time step.
        title (str): Title of the heatmap.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(clustering_scores, cmap="viridis", aspect="auto")
    plt.colorbar(label="Clustering Score")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Model Runs")
    plt.tight_layout()
    plt.show()


def plot_mean_with_variability(
    models, attribute: str, title: str, ylabel: str, kind: str = "std"
):
    """Plot mean line with shaded variability from a batch of models.

    Args:
        models (list): List of BaseModel instances.
        attribute (str): e.g., "agent_action_history"
        title (str): Plot title
        ylabel (str): Y-axis label
        kind (str): "std" for ±1 SD, "percentile" for 10–90% range
    """
    num_steps = len(models[0].__getattribute__(attribute))
    data_per_step = []

    for t in range(num_steps):
        grids = np.array([model.__getattribute__(attribute)[t] for model in models])
        step_means = np.array([np.mean(grid) for grid in grids])
        data_per_step.append(step_means)

    data_array = np.array(data_per_step)  # shape: (time, runs)
    time = np.arange(num_steps)
    mean = np.mean(data_array, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(time, mean, label="Mean", color="blue")

    if kind == "std":
        std = np.std(data_array, axis=1)
        plt.fill_between(
            time, mean - std, mean + std, color="blue", alpha=0.3, label="±1 Std Dev"
        )
    elif kind == "percentile":
        lower = np.percentile(data_array, 10, axis=1)
        upper = np.percentile(data_array, 90, axis=1)
        plt.fill_between(
            time, lower, upper, color="blue", alpha=0.3, label="10–90% Range"
        )

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

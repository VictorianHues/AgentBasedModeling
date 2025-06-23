"""Batch run tools for ABM project."""

import numpy as np
from scipy.ndimage import convolve

from abm_project.oop_model import BaseModel


def run_parameter_batch(
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
        model = model_class(**kwargs)
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


def local_action_agreement_score(
    flat_grid: np.ndarray, width: int, height: int, radius: int = 1
) -> float:
    """Compute a spatial clustering score based on local action agreement.

    Args:
        flat_grid (np.ndarray): 1D array of agent actions.
        width (int): Width of the agent grid.
        height (int): Height of the agent grid.
        radius (int): Neighborhood radius.

    Returns:
        float: Clustering score (0 to 1).
    """
    if flat_grid.size != width * height:
        raise ValueError("flat_grid size does not match width × height")

    grid = flat_grid.reshape((height, width))

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=int)
    kernel[radius, radius] = 0  # Exclude center

    score = 0.0
    for val in np.unique(grid):
        match = (grid == val).astype(int)
        same_neighbors = convolve(match, kernel, mode="constant", cval=0)
        total_neighbors = convolve(np.ones_like(grid), kernel, mode="constant", cval=0)
        local_agreement = same_neighbors / total_neighbors
        score += np.sum(local_agreement * match)

    return score / grid.size


def clustering_score_over_time(
    model, attribute: str = "action", width: int = 50, height: int = 50, radius: int = 1
) -> list[float]:
    """Compute a clustering score at each time step for a 1D action vector per step.

    Args:
        model: The model instance.
        attribute (str): Attribute name for the per-timestep 1D agent data.
        width (int): Grid width.
        height (int): Grid height.
        radius (int): Neighborhood radius.

    Returns:
        list[float]: Clustering scores per time step.
    """
    time_series = getattr(model, attribute)
    return [
        local_action_agreement_score(
            flat_grid, width=width, height=height, radius=radius
        )
        for flat_grid in time_series
    ]

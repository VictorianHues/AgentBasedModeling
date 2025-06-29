"""Batch run tools for ABM project."""

import numpy as np
from scipy.fft import rfft, rfftfreq
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
        raise ValueError("flat_grid size does not match width x height")

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


class UnionFind:
    """Union-Find data structure for efficient connectivity checks."""

    def __init__(self, size):
        """Initialize Union-Find structure with given size."""
        self.parent = np.arange(size)

    def find(self, x):
        """Find the root of the component containing x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the components containing x and y."""
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def analyze_environment_clusters_periodic(
    environment: np.ndarray,
    width: int,
    height: int,
    threshold: float = 0.5,
    diagonal: bool = False,
):
    """Analyze clusters in a 2D environment with periodic boundaries.

    Args:
        environment: 1D array of environment values per agent.
        width: Grid width.
        height: Grid height.
        threshold: Threshold to binarize the environment (default = 0.5)
        diagonal: If True, use 8-connectivity (diagonal neighbors included).
            If False, use 4-connectivity (adjacent only).

    Returns:
        num_clusters: Number of clusters found
        cluster_sizes: List of sizes of each cluster
        labels: 2D array of labeled clusters
    """
    env_grid = environment.reshape((height, width))
    binary = env_grid > threshold
    uf = UnionFind(width * height)

    def idx(x, y):
        return (y % height) * width + (x % width)  # periodic indexing

    # Define neighbor offsets
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # 4-connectivity
    if diagonal:
        neighbors += [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # 8-connectivity

    for y in range(height):
        for x in range(width):
            if not binary[y, x]:
                continue
            for dx, dy in neighbors:
                nx, ny = (x + dx) % width, (y + dy) % height
                if binary[ny, nx]:
                    uf.union(idx(x, y), idx(nx, ny))

    # Count clusters
    labels = -np.ones((height, width), dtype=int)
    label_map = {}
    next_label = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x]:
                root = uf.find(idx(x, y))
                if root not in label_map:
                    label_map[root] = next_label
                    next_label += 1
                labels[y, x] = label_map[root]

    num_clusters = len(label_map)
    cluster_sizes = [np.sum(labels == i) for i in range(num_clusters)]
    return num_clusters, cluster_sizes, labels


def get_dominant_frequency_and_power(
    signal: np.ndarray, dt: float = 1.0
) -> tuple[float, float]:
    """Calculate the dominant frequency and its power in a time series signal.

    Args:
        signal (np.ndarray): 1D array of time series data.
        dt (float): Time step between samples in seconds.

    Returns:
        tuple[float, float]: Dominant frequency (Hz) and its power.
    """
    signal = signal - signal.mean()
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=dt)
    if len(spectrum) <= 1:
        return 0.0, 0.0
    dominant_idx = np.argmax(spectrum[1:]) + 1
    return freqs[dominant_idx], spectrum[dominant_idx] ** 2

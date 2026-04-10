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


def compute_spatial_correlation(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial spatial correlation C(r) for a 2D field with periodic BCs.

    Correlation is computed from mean-subtracted values and normalised so C(0)=1.
    A fast FFT-based autocovariance is used before radial averaging.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")

    height, width = grid.shape
    field = grid.astype(np.float64, copy=False)
    fluctuations = field - field.mean()

    # Periodic autocovariance from Wiener-Khinchin theorem
    spectrum = np.fft.fft2(fluctuations)
    autocov = np.fft.ifft2(np.abs(spectrum) ** 2).real / (height * width)

    c0 = autocov[0, 0]
    max_radius = min(height, width) // 2
    distances = np.arange(max_radius + 1, dtype=np.float64)
    if c0 <= 0:
        corr_values = np.full_like(distances, np.nan)
        corr_values[0] = 1.0
        return distances, corr_values

    y_idx = np.arange(height)
    x_idx = np.arange(width)
    dy = np.minimum(y_idx, height - y_idx)[:, None]
    dx = np.minimum(x_idx, width - x_idx)[None, :]
    radial_distance = np.sqrt(dx**2 + dy**2)

    radial_bin = np.rint(radial_distance).astype(np.int64)
    mask = radial_bin <= max_radius

    binned_sum = np.bincount(
        radial_bin[mask].ravel(),
        weights=autocov[mask].ravel(),
        minlength=max_radius + 1,
    )
    binned_count = np.bincount(radial_bin[mask].ravel(), minlength=max_radius + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr_values = binned_sum / binned_count
    corr_values = corr_values / c0
    corr_values[0] = 1.0

    return distances, corr_values


def estimate_correlation_length(
    distances: np.ndarray,
    correlation_values: np.ndarray,
) -> float:
    """Estimate correlation length from C(r).

    Uses first crossing below exp(-1). If no crossing exists, falls back to
    exponential decay fit in log space on positive correlations.
    """
    dist = np.asarray(distances, dtype=np.float64)
    corr = np.asarray(correlation_values, dtype=np.float64)
    if dist.size == 0 or corr.size == 0:
        return float("nan")

    valid = np.isfinite(dist) & np.isfinite(corr)
    dist = dist[valid]
    corr = corr[valid]
    if dist.size < 2:
        return float("nan")

    target = np.exp(-1)

    positive_distance = dist > 0
    d = dist[positive_distance]
    c = corr[positive_distance]
    if d.size == 0:
        return float("nan")

    below = np.where(c <= target)[0]
    if below.size > 0:
        i = int(below[0])
        if i == 0:
            return float(d[0])

        d0, d1 = d[i - 1], d[i]
        c0, c1 = c[i - 1], c[i]
        if c1 == c0:
            return float(d1)

        frac = (target - c0) / (c1 - c0)
        return float(d0 + frac * (d1 - d0))

    # Fallback: fit log(C(r)) = a + b r on positive C(r)
    fit_mask = (d > 0) & (c > 0)
    if np.count_nonzero(fit_mask) < 2:
        return float("nan")

    x = d[fit_mask]
    y = np.log(c[fit_mask])
    slope, _intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return float("nan")

    return float(-1.0 / slope)


def compute_spatial_variance(env: np.ndarray) -> float:
    """Compute spatial variance across agents for one timestep.

    Var(n) = E[n^2] - (E[n])^2
    """
    values = np.asarray(env, dtype=np.float64).ravel()
    if values.size == 0:
        return float("nan")

    mean_n = values.mean()
    mean_n_sq = np.mean(values**2)
    return float(mean_n_sq - mean_n**2)


def compute_time_averaged_variance(env_timeseries: np.ndarray, last_k: int) -> float:
    """Compute average spatial variance over the last k timesteps."""
    series = np.asarray(env_timeseries, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(
            "env_timeseries must be a 2D array of shape (time, num_agents)"
        )
    if series.shape[0] == 0:
        return float("nan")

    k = max(1, min(int(last_k), series.shape[0]))
    window = series[-k:]
    variances = np.apply_along_axis(compute_spatial_variance, 1, window)
    return float(np.nanmean(variances))


def compute_autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized temporal autocorrelation C(tau) up to max_lag.

    C(tau) = <x_t x_{t+tau}> - <x>^2, normalised so C(0)=1.
    Uses an FFT-based autocovariance for efficiency.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.float64)

    n = x.size
    lag_max = max(0, min(int(max_lag), n - 1))

    x = x - x.mean()
    nfft = 1 << ((2 * n - 1).bit_length())
    spectrum = np.fft.rfft(x, n=nfft)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:n]

    # Unbiased covariance estimate at each lag.
    autocov = autocov / np.arange(n, 0, -1)
    c0 = autocov[0]
    if c0 <= 0 or not np.isfinite(c0):
        acf = np.zeros(lag_max + 1, dtype=np.float64)
        acf[0] = 1.0
        return acf

    acf = autocov[: lag_max + 1] / c0
    acf[0] = 1.0
    return acf


def estimate_autocorrelation_time(acf: np.ndarray) -> float:
    """Estimate characteristic autocorrelation decay time from ACF.

    Uses first crossing below exp(-1); falls back to exponential fit in log-space.
    """
    corr = np.asarray(acf, dtype=np.float64)
    if corr.size < 2:
        return float("nan")

    lags = np.arange(corr.size, dtype=np.float64)
    valid = np.isfinite(corr)
    corr = corr[valid]
    lags = lags[valid]
    if corr.size < 2:
        return float("nan")

    target = np.exp(-1)
    positive = lags > 0
    c = corr[positive]
    t = lags[positive]
    if c.size == 0:
        return float("nan")

    below = np.where(c <= target)[0]
    if below.size > 0:
        i = int(below[0])
        if i == 0:
            return float(t[0])
        t0, t1 = t[i - 1], t[i]
        c0, c1 = c[i - 1], c[i]
        if c1 == c0:
            return float(t1)
        frac = (target - c0) / (c1 - c0)
        return float(t0 + frac * (t1 - t0))

    fit_mask = (t > 0) & (c > 0)
    if np.count_nonzero(fit_mask) < 2:
        return float("nan")

    slope, _intercept = np.polyfit(t[fit_mask], np.log(c[fit_mask]), 1)
    if slope >= 0:
        return float("nan")

    return float(-1.0 / slope)


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

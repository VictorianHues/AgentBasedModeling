"""Implementation of the Hoshen-Kopelman algorithm for cluster analysis."""

import numpy as np
from scipy.sparse import coo_matrix, issparse


def _find(parent, x):
    # Path compression
    if parent[x] != x:
        parent[x] = _find(parent, parent[x])
    return parent[x]


def _union(parent, rank, x, y):
    # Union by rank
    root_x = _find(parent, x)
    root_y = _find(parent, y)
    if root_x == root_y:
        return
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1


def hoshen_kopelman(lattice):
    """Identify connected clusters in a 2D lattice using the Hoshen-Kopelman algorithm.

    Performs a two-pass union-find labeling on a binary 2D lattice to detect connected
    components (clusters) of occupied sites.

    Args:
        lattice: 2D numpy.ndarray or scipy.sparse matrix of shape (height, width)
            containing binary values (0 for empty, 1 for occupied). If a sparse
            matrix is provided, it will be traversed directly to avoid dense
            conversion.

    Returns:
        labels: 2D array or sparse matrix of the same shape as `lattice`, where each
            occupied site is marked with its cluster label (an integer), and empty
            sites remain 0.
        n_clusters: int
            Total number of distinct clusters detected.
        sizes: dict[int, int]
            Mapping from each cluster label to its size
            (number of sites in that cluster).
    """
    # Determine occupied coordinates
    if issparse(lattice):
        rows, cols = lattice.nonzero()
        coords = sorted(zip(rows, cols, strict=False))
        height, width = lattice.shape
    else:
        height, width = lattice.shape
        coords = [(i, j) for i in range(height) for j in range(width) if lattice[i, j]]

    parent = {}
    rank = {}
    labels = {}
    next_label = 1

    # First pass: assign provisional labels and record unions
    for i, j in coords:
        # check neighbors: up and left
        neighbor_labels = []
        for di, dj in [(-1, 0), (0, -1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width:
                if issparse(lattice):
                    if lattice[ni, nj]:
                        neighbor_labels.append(labels[(ni, nj)])
                else:
                    if lattice[ni, nj]:
                        neighbor_labels.append(labels[(ni, nj)])
        if not neighbor_labels:
            # new cluster
            parent[next_label] = next_label
            rank[next_label] = 0
            labels[(i, j)] = next_label
            next_label += 1
        else:
            # adopt smallest root and union others
            roots = [_find(parent, lab) for lab in neighbor_labels]
            min_root = min(roots)
            labels[(i, j)] = min_root
            for lab in roots:
                _union(parent, rank, min_root, lab)

    # Second pass: flatten labels and count sizes
    sizes = {}
    for coord, lab in labels.items():
        root = _find(parent, lab)
        labels[coord] = root
        sizes[root] = sizes.get(root, 0) + 1
    n_clusters = len(sizes)

    # Build output label structure
    if issparse(lattice):
        data = [labels[c] for c in coords]
        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]
        label_mat = coo_matrix((data, (rows, cols)), shape=(height, width)).tocsr()
        return label_mat, n_clusters, sizes
    else:
        label_array = np.zeros_like(lattice, dtype=int)

    for (i, j), lab in labels.items():
        label_array[i, j] = lab

    return label_array, n_clusters, sizes


def cluster_time_series(model, option: str = "action"):
    """Run Hoshen-Kopelman on each slice of the action or environment history.

    Args:
        model : BaseModel
            The model instance containing the action history.
        option : str, optional
            The option to choose between "action" or "environment"
            history.

    Returns:
        nc : np.ndarray, shape (T,)
            number of clusters at each time step
        c_max : np.ndarray, shape (T,)
            size of the largest cluster at each time step
    """
    if option == "action":
        history = model.action[: model.time + 1]
        # print("Action History:", history.shape)
    elif option == "environment":
        history = model.environment[: model.time + 1]
        # print("Environment History:", history.shape)
    else:
        raise ValueError(f"Unknown option: {option}")

    history = history.reshape((-1, model.height, model.width))
    T = history.shape[0]

    nc = np.zeros(T, dtype=int)
    c_max = np.zeros(T, dtype=int)

    for t in range(T):
        lattice = history[t]
        if option == "action":
            lattice = np.where(lattice == -1, 0, lattice)
        elif option == "environment":
            # Using 0.6 as threshold for "good" environment
            lattice = np.where(lattice > 0.6, 1, 0)

        if issparse(lattice):
            lattice = lattice.toarray()

        labels, n_clusters, sizes = hoshen_kopelman(lattice)
        nc[t] = n_clusters
        c_max[t] = max(sizes.values(), default=0)

    return labels, nc, c_max


def cluster_given_timestep(model, option: str = "action", timestep: int = 0):
    """Run Hoshen-Kopelman on a specific timestep of the action or environment history.

    Args:
        model : BaseModel
            The model instance containing the action history.
        option : str, optional
            The option to choose between "action" or "environment" history.
        timestep : int, optional
            The specific timestep to analyze.

    Returns:
        labels : np.ndarray or scipy.sparse.coo_matrix
            Labeled clusters at the specified timestep.
        n_clusters : int
            Number of clusters detected at the specified timestep.
        sizes : dict[int, int]
            Mapping from each cluster label to its size
            (number of sites in that cluster).
    """
    # if timestep > model.num_steps:
    #     raise ValueError(f"Timestep {timestep} exceeds {model.num_steps}")

    environment_threshold = 0.6  # can change this later

    if option == "action":
        history = model.action[: model.time + 1]
        # print("Action History:", history.shape)
    elif option == "environment":
        history = model.environment[: model.time + 1]
        # print("Environment History:", history.shape)
    else:
        raise ValueError(f"Unknown option: {option}")

    history = history.reshape((-1, model.height, model.width))
    # T = history.shape[0]

    lattice = history[timestep]

    if option == "action":
        lattice = np.where(lattice == -1, 0, lattice)
    elif option == "environment":
        # Using 0.6 as threshold for "good" environment
        lattice = np.where(lattice > environment_threshold, 1, 0)

    # if issparse(lattice):
    #     lattice = lattice.toarray()

    labels, n_clusters, sizes = hoshen_kopelman(lattice)

    return n_clusters, sizes


def correlation_length_snapshot(label_array: np.ndarray, sizes: dict[int, int]):
    """Estimate correlation length for a 2d array using the FK second-moment method.

    Args:
        label_array: np.ndarray
            2-D integer array of shape (height, width).
            Output of `hoshen_kopelman`, where occupied sites carry a
            positive cluster label and empty sites are 0.
        sizes: dict[int, int]
            Mapping {cluster_label → cluster_size} for the same snapshot.

    Returns:
        float: Correlation length ξ for the snapshot.
               Returns 0.0 if no cluster has size ≥ 2.
    """
    numerator = 0.0
    denominator = 0.0

    for lab, s in sizes.items():
        if s < 2:  # singleton -> Rg undefined
            continue

        ys, xs = np.where(label_array == lab)  # coords of cluster
        coords = np.stack((ys, xs), axis=1).astype(float)
        r_cm = coords.mean(axis=0)  # centre of mass
        rg2 = ((coords - r_cm) ** 2).sum(axis=1).mean()  # ⟨r²⟩

        numerator += (s**2) * rg2
        denominator += s

    return 0.0 if denominator == 0.0 else np.sqrt(numerator / denominator)


def correlation_length_time_series(model, option: str = "action"):
    """Compute correlation length and basic cluster statistics.

    Args:
        model: BaseModel
            Instance with attributes `action` and `environment`, which are
            3D numpy arrays of shape (T, height, width), where T is the
            number of time steps (model.time + 1).
        option: str
            Either "action" or "environment".  Selects which lattice
            history to analyse.

    Returns:
        dict[str, np.ndarray]:
            {
                "xi":               Correlation length ξ(t), shape (T,),
                "num_clusters":     Number of clusters at each t, shape (T,),
                "max_cluster_size": Size of the largest cluster at each t,
                                    shape (T,),
                "mean_cluster_size":Mean cluster size at each t, shape (T,)
            }
            where T = model.time + 1.
    """
    if option == "action":
        history = model.action[: model.time + 1]
    elif option == "environment":
        history = model.environment[: model.time + 1]
    else:
        raise ValueError(f"Unknown option: {option}")

    history = history.reshape((-1, model.height, model.width))
    T = history.shape[0]

    xi = np.zeros(T, dtype=float)
    num_clusters = np.zeros(T, dtype=int)
    max_cluster_size = np.zeros(T, dtype=int)
    mean_cluster_sz = np.zeros(T, dtype=float)

    for t in range(T):
        lattice = history[t]

        if option == "action":
            lattice = np.where(lattice == -1, 0, lattice)  # defect=0
        else:  # environment
            lattice = np.where(lattice > 0.6, 1, 0)  # threshold

        if issparse(lattice):
            lattice = lattice.toarray()

        labels, n_clusters, sizes = hoshen_kopelman(lattice)

        xi[t] = correlation_length_snapshot(labels, sizes)
        num_clusters[t] = n_clusters
        max_cluster_size[t] = max(sizes.values(), default=0)
        mean_cluster_sz[t] = np.mean(list(sizes.values())) if sizes else 0.0

    # return xi, num_clusters, max_cluster_size, mean_cluster_sz
    return {
        "xi": xi,
        "num_clusters": num_clusters,
        "max_cluster_size": max_cluster_size,
        "mean_cluster_size": mean_cluster_sz,
    }

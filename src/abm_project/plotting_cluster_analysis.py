"""Plotting functions for cluster analysis in agent-based models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from abm_project.cluster_analysis import cluster_time_series


def plot_ncluster_single_model(model, option, savedir):
    """Plot the number of clusters for a single model over time.

    Args:
        model (VectorisedModel): The model instance to analyze.
        option (str): The option to choose between "action" or "environment" history.
        savedir (Path): Directory to save the plot.

    Returns:
        None: Saves the plot as a PNG file.
    """
    savedir = savedir or Path(".")

    _, nc, c1 = cluster_time_series(model=model, option=option)
    print(f"Number of clusters at each time step: {nc}")
    print(f"Largest cluster fraction at each time step: {c1 / model.num_agents}")

    plt.figure(figsize=(10, 6))
    plt.plot(nc, label="Number of Clusters")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Clusters")
    plt.title("Number of Clusters Over Time")
    plt.legend()
    plt.grid()

    if savedir:
        plt.savefig(
            savedir / "n_clusters.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'n_clusters.png'")
    plt.show()


# def plot_ncluster_across_memory(cluster_n, savedir):
#     savedir = savedir or Path(".")

#     memory_values = list(cluster_n.keys())

#     fig, ax = plt.figure(figsize=(10, 6))
#     for mem in memory_values:
#         ax.plot(cluster_n[mem], label=f"Memory {mem}")
#     ax.set_xlabel("Time Steps")
#     ax.set_ylabel("Number of Clusters")
#     ax.set_title("Number of Clusters Over Time for Different Memory Counts")
#     ax.legend()
#     ax.grid()
#     if savedir:
#         fig.savefig(
#             savedir / f"n_clusters_across_memory_{option}.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print("Plot saved as 'n_clusters_across_memory.png'")
#     plt.show()


def plot_eqenv_across_memory_rationality(data_file, savedir=None):
    """Plot the equilibrium environment status from a saved npz file.

    Args:
        data_file (str or Path): Path to the npz file containing equilibrium
                                    environment data.
        savedir (Path, optional): Directory to save the plot.
                                    Defaults to current directory.

    Returns:
        None: Saves the plot as a PNG file.
    """
    savedir = savedir or Path(".")

    data = np.load(data_file)
    eq_env_status = data["eq_env_status"]
    memory_range = data["memory_range"]
    rationality_range = data["rationality_range"]

    # Plotting the equilibrium environment status
    fig, ax = plt.subplots(figsize=(4.5, 4.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    cbar = ax.imshow(
        eq_env_status,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[
            rationality_range[0],
            rationality_range[-1],
            memory_range[0],
            memory_range[-1],
        ],
        vmax=1.0,
        vmin=0.0,
    )
    plt.colorbar(cbar, label="Equilibrium Environment Status")
    plt.xlabel("Rationality Level", fontsize=12)
    plt.ylabel("Memory Count", fontsize=12)
    plt.xticks(rationality_range, fontsize=10)
    plt.xticks(rotation=90)
    # Show only 6 ticks on the x-axis
    plt.yticks(memory_range, fontsize=10)
    # plt.title("Equilibrium Environment Across Memory and Rationality")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()

    if savedir:
        fig.savefig(
            savedir / "eq_env_across_memory_rationality_lowgs.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'eq_env_across_memory_rationality_lowgs.pdf'")

    # plt.show()
    plt.close(fig)


def plot_cluster_across_memory_rationality(data_file=None, savedir=None):
    """Plot the number of clusters across different memory and rationality levels.

    Args:
        data_file (str or Path): Path to the npz file containing cluster data.
        savedir (Path, optional): Directory to save the plot.
                                    Defaults to current directory.

    Returns:
        None: Saves the plot as a PNG file.
    """
    savedir = savedir or Path(".")

    data = np.load(data_file)
    ncluster = data["ncluster"]
    memory_range = data["memory_range"]
    rationality_range = data["rationality_range"]

    # Plotting the number of clusters
    fig, ax = plt.subplots(figsize=(5, 3.75))
    cbar = ax.imshow(
        ncluster,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[
            rationality_range[0],
            rationality_range[-1],
            memory_range[0],
            memory_range[-1],
        ],
        vmax=np.max(ncluster),
        vmin=np.min(ncluster),
    )
    plt.colorbar(cbar, label="Number of Clusters")
    plt.xlabel("Rationality Level")
    plt.ylabel("Memory Count")
    plt.xticks(rationality_range)
    plt.xticks(rotation=90)
    plt.yticks(memory_range)
    plt.title("Number of Clusters Across Memory and Rationality")
    plt.tight_layout()

    if savedir:
        fig.savefig(
            savedir / "n_clusters_across_memory_rationality.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'n_clusters_across_memory_rationality.png'")

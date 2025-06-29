"""Script to plot cluster analysis results."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abm_project.cluster_analysis import (
    cluster_time_series,
)


def plot_eq_env_against_memory_rationality(filepath, savedir=None):
    """Create a heatmap of memory vs rationality filled with equiblibrium mean env.

    Args:
        filepath: Path to the npz file containing equilibrium environment state.
        savedir: Directory to save the plot. If None, saves in current directory.
    """
    # Read the equilibrium state from the npz file
    data = np.load(filepath)
    eq_env_state = data["eq_env_state"]
    memory_range = data["memory_range"]
    rat_range = data["rat_range"]

    plt.figure(figsize=(12, 8))
    # plot equilibrium env state (environment state at last timestep)
    plt.imshow(
        eq_env_state,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[rat_range[0], rat_range[-1], memory_range[0], memory_range[-1]],
        vmax=np.max(1.0),
        vmin=np.min(0.0),
    )

    plt.colorbar(label="Equilibrium Environment")
    plt.xlabel("Rationality (rat)")
    plt.ylabel("Memory Count (mem_count)")
    # plt.title("Steady-State Environment as Function of rat and mem_count")
    plt.xticks(rat_range)
    plt.xticks(rotation=90)
    plt.yticks(memory_range)
    plt.tight_layout()

    if savedir:
        savedir += (
            f"memory_({memory_range[0]},{memory_range[-1]})"
            + f"rationality({rat_range[0]},{rat_range[-1]})"
        )
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(
            os.path.join(savedir, "eq_env_vs_memory_rationality.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'eq_env_vs_memory_rationality.png'")
    # plt.show()
    plt.close()


def plot_nclusters_against_memory_rationality(
    filepath, savedir=None, option="environment"
):
    """Plot the number of clusters against memory and rationality.

    Args:
        filepath: Path to the npz file containing cluster data.
        savedir: Directory to save the plot. If None, saves in current directory.
        option: Option indicating the type of analysis (e.g., "environment").
    """
    # Read the equilibrium state from the npz file
    data = np.load(filepath)
    n_clusters = data["n_clusters"]
    memory_range = data["memory_range"]
    rat_range = data["rat_range"]

    plt.figure(figsize=(12, 8))
    plt.imshow(
        n_clusters,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[rat_range[0], rat_range[-1], memory_range[0], memory_range[-1]],
        vmax=np.max(1.0),
        vmin=np.min(0.0),
    )

    plt.colorbar(label="Number of Clusters")
    plt.xlabel("Rationality (rat)")
    plt.ylabel("Memory Count (mem_count)")
    plt.title(f"Number of Clusters as Function of rat and mem_count ({option})")
    plt.xticks(rat_range)
    plt.xticks(rotation=90)
    plt.yticks(memory_range)

    if savedir:
        savedir += (
            f"memory_({memory_range[0]},{memory_range[-1]})"
            + f"rationality({rat_range[0]},{rat_range[-1]})"
        )
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(
            os.path.join(savedir, "n_clusters_vs_memory.png"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_ncluster_across_memory(filepath, cluster_n, option, savedir):
    """Plot the number of clusters across different memory counts.

    Args:
        filepath: Path to the npz file containing cluster data.
        cluster_n: Dictionary where keys are memory counts and values are lists of
                    number of clusters at each time step.
        option: Option indicating the type of analysis (e.g., "environment").
        savedir: Directory to save the plot. If None, saves in current directory.
    """
    savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    memory_values = list(cluster_n.keys())

    fig, ax = plt.figure(figsize=(10, 6))
    for mem in memory_values:
        ax.plot(cluster_n[mem], label=f"Memory {mem}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Number of Clusters Over Time for Different Memory Counts")
    ax.legend()
    ax.grid()
    if savedir:
        fig.savefig(
            savedir / f"n_clusters_across_memory_{option}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'n_clusters_across_memory.png'")
    # plt.show()
    plt.close()


def plot_ncluster_multiple_model_runs(num_steps=1000, models=None, savedir=None):
    """Plot the average number of clusters across multiple model runs.

    Args:
        num_steps: Number of timesteps to run the model.
        models: List of VectorisedModel instances to analyze.
        savedir: Directory to save the plot. If None, saves in current directory.
    """
    if models is None:
        raise ValueError("No models provided for plotting.")

    savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    num_clusters = []
    max_cluster_sizes = []
    for model in models:
        nc, c1 = cluster_time_series(model=model, option="environment")
        num_clusters.append(nc)
        max_cluster_sizes.append(c1)

    # Calculate the average number of clusters and max cluster sizes across all models
    avg_num_clusters = np.mean(num_clusters, axis=0)
    avg_max_cluster_sizes = np.mean(max_cluster_sizes, axis=0)

    # Calculate the standard deviation for error bars
    std_num_clusters = np.std(num_clusters, axis=0)
    std_max_cluster_sizes = np.std(max_cluster_sizes, axis=0)

    # Plot the average number of clusters
    timesteps = np.arange(len(avg_num_clusters))
    plt.figure(figsize=(10, 6))

    # Plot mean and shaded std of number of clusters
    plt.plot(timesteps, avg_num_clusters, label="Mean Num. of Clusters", color="blue")
    plt.fill_between(
        timesteps,
        avg_num_clusters - std_num_clusters,
        avg_num_clusters + std_num_clusters,
        alpha=0.3,
        label="±1 Std Dev (# of Clusters)",
        color="blue",
    )

    # Plot mean and shaded std of max cluster sizes
    plt.plot(
        timesteps, avg_max_cluster_sizes, label="Mean Max Cluster Size", color="orange"
    )
    plt.fill_between(
        timesteps,
        avg_max_cluster_sizes - std_max_cluster_sizes,
        avg_max_cluster_sizes + std_max_cluster_sizes,
        alpha=0.3,
        label="±1 Std Dev (Max Cluster Size)",
        color="orange",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Number of Clusters")
    plt.title("Number of Clusters Over Time Across Runs")
    plt.legend()
    plt.grid(True)

    if savedir:
        plt.savefig(
            savedir
            / f"average_num_clusters_over_time_mem{models[0].memory_count}_\
            rat{models[0].rationality}_gs{models[0].gamma_s}_{num_steps}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'average_num_clusters_over_time.png'")

    # plt.show()
    plt.close()


def plot_ncluster_given_memory(filepath, model, option, num_steps, savedir):
    """Plot the number of clusters over time for a given model and option.

    Args:
        filepath: Path to the npz file containing cluster data.
        model: VectorisedModel instance to analyze.
        option: Option indicating the type of analysis (e.g., "environment").
        num_steps: Number of timesteps to run the model.
        savedir: Directory to save the plot. If None, saves in current directory.
    """
    savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    nc, c1 = cluster_time_series(model=model, option=option)

    plt.figure(figsize=(10, 6))
    plt.plot(nc, label="Number of Clusters")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Clusters")
    plt.title("Number of Clusters Over Time")
    plt.legend()
    plt.grid()

    if savedir:
        plt.savefig(
            savedir
            / f"mem{model.memory_count}_{option}_{model.rationality:.3f}_\
            gammas{model.gamma_s}_{model.width}_{model.height}_{num_steps}.png",
            dpi=300,
            bbox_inches="tight",
        )
    # plt.show()
    plt.close()

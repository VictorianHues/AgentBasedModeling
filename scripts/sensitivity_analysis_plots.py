from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm

from abm_project.plotting import configure_mpl


def plot_outcome_distributions(
    mean_environment, mean_action, pluralistic_ignorance, cluster_count, savedir: Path
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(3, 2),
        constrained_layout=True,
    )

    # Mean environment
    axes[0, 0].hist(
        mean_environment,
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    axes[0, 0].set_title("Mean environment")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Mean action
    axes[0, 1].hist(
        mean_action,
        bins=30,
        color="darkorange",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    axes[0, 1].set_title("Mean action")
    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Average pluralistic ignorance
    axes[1, 0].hist(
        pluralistic_ignorance,
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    axes[1, 0].set_title("Pluralistic ignorance")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Number of clusters
    axes[1, 1].hist(
        cluster_count, bins=30, color="gold", edgecolor="black", alpha=0.7, density=True
    )
    axes[1, 1].set_title("Cluster count")
    axes[1, 1].set_xlim(0, None)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].set_yscale("log")

    fig.supylabel("Density")

    fig.savefig(
        savedir / "sensitivity_analysis_outcome_distributions.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")
    outcomes = np.load(DATA_DIR / "sensitivity_analysis_outcome_measurements.npz")
    parameters = outcomes["parameters"]
    mean_environment = outcomes["mean_environment"]
    mean_action = outcomes["mean_action"]
    pluralistic_ignorance = outcomes["pluralistic_ignorance"]
    cluster_count = outcomes["cluster_count"]

    configure_mpl()
    plot_outcome_distributions(
        mean_environment, mean_action, pluralistic_ignorance, cluster_count, FIGURES_DIR
    )
    # plot_pawn_heatmap()
    # plot_sobol_indices()

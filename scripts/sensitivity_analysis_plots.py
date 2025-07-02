import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm as tqdm

from abm_project import sensitivity_analysis
from abm_project.plotting import configure_mpl
from abm_project.sensitivity_analysis import (
    PawnIndices,
    SobolIndices,
    pawn_analysis,
    sobol_analysis,
)


def plot_outcome_distributions(
    mean_environment,
    mean_action,
    pluralistic_ignorance,
    cluster_count,
    peak_frequency,
    dominant_frequency_power,
    savedir: Path,
    quality_label: str,
):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(3, 3),
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

    # Peak frequency
    axes[2, 0].hist(
        peak_frequency,
        bins=30,
        color="gold",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    axes[2, 0].set_title("Peak frequency")
    axes[2, 0].set_xlim(0, None)
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    axes[2, 0].set_yscale("log")

    # Dominant frequency power
    axes[2, 1].hist(
        dominant_frequency_power,
        bins=30,
        color="gold",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    axes[2, 1].set_title("Dominant freq. power")
    axes[2, 1].set_xlim(0, None)
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    axes[2, 1].set_yscale("log")

    fig.supylabel("Density")

    fig.savefig(
        savedir
        / f"sensitivity_analysis_outcome_distributions_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


def plot_pawn_heatmap(indices: PawnIndices, savedir: Path, quality_label: str):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    sns.heatmap(
        indices.stack(),
        vmin=0,
        vmax=0.5,
        square=True,
        annot=True,
        linewidths=1,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.4},
        cmap="rocket_r",
        yticklabels=sensitivity_analysis.OUTCOME_NAMES,
        xticklabels=sensitivity_analysis.PARAMETER_NAMES,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(
        savedir / f"pawn_heatmap_{quality_label}_quality.pdf", bbox_inches="tight"
    )


def plot_sobol_indices(indices: SobolIndices, savedir: Path, quality_label: str):
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(3.5, 1.75),
        constrained_layout=True,
        sharey=True,
    )

    n_indices = len(sensitivity_analysis.PARAMETER_NAMES)
    n_outcomes = len(indices.to_list())

    y = np.arange(n_indices)

    # If plotting multiple outcomes, offset each one
    if n_outcomes == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.25, 0.25, n_outcomes, endpoint=True)

    for idxes, offset in zip(indices.to_list(), offsets, strict=True):
        # First-order
        axes[0].errorbar(
            idxes.first_order.index,
            y - offset,
            xerr=idxes.first_order.confidence,
            linestyle="None",
            marker="o",
            linewidth=0.5,
            markersize=2,
        )

        # Total-order
        axes[1].errorbar(
            idxes.total_order.index,
            y - offset,
            xerr=idxes.total_order.confidence,
            linestyle="None",
            marker="o",
            linewidth=0.5,
            markersize=2,
        )

    axes[0].set_ylim(-0.3, n_indices - 1 + 0.3)
    axes[0].set_yticks(range(n_indices), sensitivity_analysis.PARAMETER_NAMES)

    axes[0].set_title("First-order")
    axes[0].set_xlabel(r"$S_i$")
    axes[1].set_title("Total-order")
    axes[1].set_xlabel(r"$S_T$")

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)

    fig.savefig(
        savedir / f"sobol_indices_{quality_label}_quality.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    outcomes = np.load(
        DATA_DIR
        / f"sensitivity_analysis_outcome_measurements_{quality_label}_quality.npz"
    )
    parameters = outcomes["parameters"]
    mean_environment = outcomes["mean_environment"]
    mean_action = outcomes["mean_action"]
    pluralistic_ignorance = outcomes["pluralistic_ignorance"]
    cluster_count = outcomes["cluster_count"]
    peak_frequency = outcomes["peak_frequency"]
    dominant_frequency_power = outcomes["dominant_frequency_power"]

    pawn_indices = pawn_analysis(
        parameters,
        mean_environment,
        mean_action,
        pluralistic_ignorance,
        cluster_count,
        peak_frequency,
        dominant_frequency_power,
    )
    sobol_indices = sobol_analysis(
        pluralistic_ignorance,
    )

    configure_mpl()
    plot_pawn_heatmap(pawn_indices, FIGURES_DIR, quality_label)
    plot_sobol_indices(sobol_indices, FIGURES_DIR, quality_label)
    plot_outcome_distributions(
        mean_environment,
        mean_action,
        pluralistic_ignorance,
        cluster_count,
        peak_frequency,
        dominant_frequency_power,
        FIGURES_DIR,
        quality_label,
    )

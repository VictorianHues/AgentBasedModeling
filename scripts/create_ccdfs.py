import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from tqdm import tqdm

from abm_project.batch_run_tools import (
    analyze_environment_clusters_periodic,
    get_dominant_frequency_and_power,
)
from abm_project.plotting import configure_mpl
from abm_project.utils import linear_update, piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def run_model(args):
    (
        r,
        i,
        lmbda,
        gamma_s,
        neighb_prediction_option,
        severity_benefit_option,
        radius_option,
        b_2,
        env_update_fn_type,
        memory_count,
    ) = args
    num_agents = 900
    width = 30
    height = 30
    num_steps = 1000

    if env_update_fn_type == "linear":
        env_update_fn = linear_update(0.01)
    elif env_update_fn_type == "piecewise":
        env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    else:
        raise ValueError(f"Unknown env_update_fn_type: {env_update_fn_type}")

    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=None,
        env_update_fn=env_update_fn,
        rationality=lmbda,
        simmer_time=1,
        neighb_prediction_option=neighb_prediction_option,
        severity_benefit_option=severity_benefit_option,
        radius_option=radius_option,
        max_storage=num_steps,
        b_1=np.full(num_agents, 1.0),
        b_2=b_2 if b_2 is not None else None,
        gamma_s=gamma_s,
    )
    model.run(num_steps)

    final_env = model.environment[model.time]
    mean_env = final_env.mean()

    num_clusters, cluster_sizes, _ = analyze_environment_clusters_periodic(
        final_env, width, height, threshold=0.5
    )
    mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    max_cluster_size = np.max(cluster_sizes) if cluster_sizes else 0

    env_timeseries = model.environment[: model.time].mean(axis=1)
    peak_freq, power = get_dominant_frequency_and_power(env_timeseries)

    return (
        r,
        i,
        num_clusters,
        mean_cluster_size,
        max_cluster_size,
        mean_env,
        peak_freq,
        power,
    )


def plot_distributions_for_param_combo(
    lmbda: float,
    gamma_s: float,
    neighb: str = "linear",
    severity: str = "adaptive",
    radius: str = "single",
    b2: np.ndarray = None,
    env_update_type: str = "linear",
    repeats: int = 100,
    memory_count: int = 10,
    seed: int = 42,
    savedir: Path = Path("results/figures"),
    quality_label: str = "low",
):
    np.random.seed(seed)
    random.seed(seed)

    results = defaultdict(list)

    tasks = [
        (
            r,
            r,
            lmbda,
            gamma_s,
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            memory_count,
        )
        for r in range(repeats)
    ]

    for args in tqdm(tasks, desc="Running simulations"):
        try:
            _, _, num_clusters, _, _, mean_env, peak_freq, power = run_model(args)
            results["num_clusters"].append(num_clusters)
            results["mean_env"].append(mean_env)
            results["peak_freq"].append(peak_freq)
            results["power"].append(power)
        except Exception as e:
            print(f"[!] Error on run {args[0]}: {e}")

    plot_info = [
        ("num_clusters", "Cluster Count", "Clusters"),
        ("mean_env", "Mean Environmental Status", "Mean Env"),
        ("peak_freq", "Dominant Frequency", "Frequency"),
        ("power", "Fourier Power", "Power"),
    ]

    # Use smaller figure size and font for double-column compatibility
    # plt.rcParams.update(
    #    {
    #        "font.size": 10,
    #        "axes.titlesize": 12,
    #        "axes.labelsize": 10,
    #        "legend.fontsize": 10,
    #        "xtick.labelsize": 10,
    #        "ytick.labelsize": 10,
    #    }
    # )

    for key, title, xlabel in plot_info:
        data = np.array(results[key])
        # Remove NaN and infinite values first
        data = data[np.isfinite(data)]
        data = data[data > 0]  # Filter zeros and negatives

        # Power-law fit using powerlaw package
        if len(data) > 10 and np.max(data) > np.min(data):  # Check for valid data range
            try:
                fit = powerlaw.Fit(data, discrete=False, verbose=False)
                alpha = fit.power_law.alpha
                xmin = fit.power_law.xmin

                # Compare to exponential distribution
                R, p = fit.distribution_compare("power_law", "exponential")

                # Plot CCDF
                fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
                fit.plot_ccdf(label="Empirical", ax=ax, color="black", linewidth=1)
                fit.power_law.plot_ccdf(
                    label=f"Power law (α={alpha:.2f})",
                    ax=ax,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                )

                # Add vertical line for x_min
                # ax.axvline(xmin,
                #    color="blue",
                #    linestyle=":",
                #    linewidth=1,
                #    label=f"$x_{{min}}$={xmin:.2f}")

                if np.min(data) > 0 and np.max(data) > np.min(data):
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                ax.set_xlabel(xlabel)
                ax.set_ylabel("CCDF")
                ax.set_title(
                    f"{title}\n($\\lambda$={lmbda:.2f}, $\\gamma_s$={gamma_s:.3f})",
                    pad=2,
                )
                ax.legend(loc="lower left", frameon=False)
                ax.text(
                    0.97,
                    0.03,
                    f"$x_{{min}}$={xmin:.2f}\nR={R:.2f}, p={p:.3f}",
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5
                    ),
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fname = (
                    f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}"
                    f"_{env_update_type}_ccdf_{quality_label}_quality.pdf"
                )
                plt.savefig(savedir / fname, bbox_inches="tight")

            except Exception as e:
                print(f"Error fitting power-law for {key}: {e}")
                # Create a simple histogram instead
                fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
                ax.hist(data, bins=20, alpha=0.7, edgecolor="black")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Frequency")
                ax.set_title(
                    f"{title}\n($\\lambda$={lmbda:.2f},$\\gamma_s$={gamma_s:.3f})",
                    pad=2,
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fname = (
                    f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}"
                    f"_{env_update_type}_hist_{quality_label}_quality.pdf"
                )
                plt.savefig(savedir / fname, bbox_inches="tight")

        else:
            print(f"Not enough valid data to fit power-law for {key} (n = {len(data)})")
            # Still create a simple plot if we have some data
            if len(data) > 0:
                fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
                ax.hist(data, bins=min(10, len(data)), alpha=0.7, edgecolor="black")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Frequency")
                ax.set_title(
                    f"{title}\n($\\lambda$={lmbda:.2f},$\\gamma_s$={gamma_s:.3f})",
                    pad=2,
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fname = (
                    f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}_"
                    f"{env_update_type}_limited_{quality_label}_quality.pdf"
                )
                plt.savefig(savedir / fname, bbox_inches="tight")

    return results


if __name__ == "__main__":
    FIGURES_DIR = Path("results/figures")
    NEIGHB = "linear"
    SEVERITY = "adaptive"
    ENV_UPDATE_TYPE = "piecewise"
    N_REPEATS = 250
    MEMORY_SIZE = 10
    PARAMS = [
        # (lambda, gamma_s)
        (4.0, 0.0042),
        (4.0, 0.0048),
    ]

    QUICK_REPEATS = 100
    FULL_REPEATS = 500

    configure_mpl()

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS

    for λ, γ in PARAMS:
        plot_distributions_for_param_combo(
            lmbda=λ,
            gamma_s=γ,
            neighb=NEIGHB,
            severity=SEVERITY,
            env_update_type=ENV_UPDATE_TYPE,
            repeats=repeats,
            memory_count=MEMORY_SIZE,
            savedir=FIGURES_DIR,
            quality_label=quality_label,
        )

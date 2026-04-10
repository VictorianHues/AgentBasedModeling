import argparse
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from tqdm import tqdm

from abm_project.batch_run_tools import (
    analyze_environment_clusters_periodic,
    compute_autocorrelation,
    compute_spatial_correlation,
    estimate_autocorrelation_time,
    estimate_correlation_length,
    get_dominant_frequency_and_power,
)
from abm_project.plotting import configure_mpl
from abm_project.utils import linear_update, piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def get_parallel_workers(default_workers: int = 6) -> int:
    """Return a conservative process count; override with ABM_MAX_WORKERS."""
    cpu_count = os.cpu_count() or 1
    env_value = os.getenv("ABM_MAX_WORKERS")
    if env_value is not None:
        try:
            return max(1, min(int(env_value), cpu_count))
        except ValueError:
            print(
                "Invalid ABM_MAX_WORKERS="
                f"'{env_value}'. Falling back to "
                f"{default_workers}."
            )
    return max(1, min(default_workers, cpu_count))


def get_chunksize(n_tasks: int, workers: int, default: int = 10) -> int:
    """Get executor map chunksize; override with ABM_CHUNKSIZE."""
    env_value = os.getenv("ABM_CHUNKSIZE")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            print(f"Invalid ABM_CHUNKSIZE='{env_value}'. Falling back to auto.")
    if n_tasks <= 0:
        return 1
    return max(1, min(default, n_tasks // max(1, workers * 8)))


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
        corr_window,
        acf_max_lag,
        return_corr_profile,
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

    rng = np.random.default_rng(r)

    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
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
    autocorrelation = compute_autocorrelation(env_timeseries, max_lag=acf_max_lag)
    autocorr_time = estimate_autocorrelation_time(autocorrelation)

    distances, corr_values, corr_length = compute_corr_profile_from_model(
        model=model,
        width=width,
        height=height,
        corr_window=corr_window,
    )

    return (
        r,
        i,
        num_clusters,
        mean_cluster_size,
        max_cluster_size,
        mean_env,
        peak_freq,
        power,
        autocorr_time,
        corr_length,
        distances if return_corr_profile else None,
        corr_values if return_corr_profile else None,
    )


def run_model_safe(args):
    """Execute one simulation and package any exception for robust batch runs."""
    try:
        return (run_model(args), None, args[0])
    except Exception as exc:
        return (None, f"{type(exc).__name__}: {exc}", args[0])


def compute_corr_profile_from_model(
    model: VectorisedModel,
    width: int,
    height: int,
    corr_window: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute C(r) and correlation length over final timesteps."""
    history = model.environment[: model.time + 1]
    window = max(1, min(corr_window, history.shape[0]))
    start = history.shape[0] - window

    distances = None
    corr_accum = []
    for t in range(start, history.shape[0]):
        grid = history[t].reshape((height, width))
        dist_t, corr_t = compute_spatial_correlation(grid)
        if distances is None:
            distances = dist_t
        corr_accum.append(corr_t)

    corr_values = np.nanmean(np.vstack(corr_accum), axis=0)
    corr_values[0] = 1.0
    corr_length = estimate_correlation_length(distances, corr_values)
    return distances, corr_values, corr_length


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
    corr_window: int = 1,
    acf_max_lag: int = 200,
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
            corr_window,
            acf_max_lag,
            False,
        )
        for r in range(repeats)
    ]

    workers = get_parallel_workers()
    chunksize = get_chunksize(n_tasks=len(tasks), workers=workers)
    print(f"Running {len(tasks)} simulations with {workers} worker process(es).")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for payload, error_message, run_id in tqdm(
            executor.map(run_model_safe, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Running simulations",
        ):
            if error_message is not None:
                print(f"[!] Error on run {run_id}: {error_message}")
                continue

            (
                _,
                _,
                num_clusters,
                mean_cluster_size,
                max_cluster_size,
                mean_env,
                peak_freq,
                power,
                autocorr_time,
                corr_length,
                _,
                _,
            ) = payload
            results["num_clusters"].append(num_clusters)
            results["mean_cluster_size"].append(mean_cluster_size)
            results["max_cluster_size"].append(max_cluster_size)
            results["mean_env"].append(mean_env)
            results["peak_freq"].append(peak_freq)
            results["power"].append(power)
            results["autocorr_time"].append(autocorr_time)
            results["corr_length"].append(corr_length)

    plot_info = [
        ("num_clusters", "Cluster Count", "Clusters"),
        ("mean_cluster_size", "Mean Cluster Size", "Mean cluster size"),
        ("max_cluster_size", "Largest Cluster Size", "Largest cluster size"),
        ("mean_env", "Mean Environmental Status", "Mean Env"),
        ("peak_freq", "Dominant Frequency", "Frequency"),
        ("power", "Fourier Power", "Power"),
        ("autocorr_time", "Autocorrelation Time", r"$\tau_c$"),
        ("corr_length", "Correlation Length", r"$\xi$"),
    ]

    figure_size = (3.5, 2)
    summary_text_size = 10

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

    for key, _title, xlabel in plot_info:
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

                # Plot both truncated and untruncated CCDFs.
                for original_data in (False, True):
                    mode_label = "Original data" if original_data else "Truncated data"
                    mode_suffix = "_original" if original_data else ""

                    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                    fit.plot_ccdf(
                        label=f"Empirical ({mode_label})",
                        ax=ax,
                        color="black",
                        linewidth=1,
                        original_data=original_data,
                    )
                    fit.power_law.plot_ccdf(
                        label=f"Power law (α={alpha:.2f})",
                        ax=ax,
                        color="red",
                        linestyle="--",
                        linewidth=1,
                    )

                    if np.min(data) > 0 and np.max(data) > np.min(data):
                        ax.set_xscale("log")
                        ax.set_yscale("log")

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("CCDF")
                    ax.legend(loc="lower left", frameon=False)
                    ax.text(
                        0.97,
                        0.03,
                        f"$x_{{min}}$={xmin:.2f}\nR={R:.2f}, p={p:.4f}",
                        transform=ax.transAxes,
                        fontsize=summary_text_size,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round",
                            facecolor="white",
                            alpha=0.7,
                            linewidth=0.5,
                        ),
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    fname = (
                        f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.4f}_neighbor_{neighb}"
                        f"_{env_update_type}_ccdf{mode_suffix}_{quality_label}_quality.pdf"
                    )
                    plt.savefig(savedir / fname, bbox_inches="tight")
                    plt.close(fig)

                    # Linear-scale companion CCDF plot from powerlaw's ccdf method
                    x_emp, ccdf_emp = fit.ccdf(original_data=original_data)
                    x_emp = np.asarray(x_emp)
                    ccdf_emp = np.asarray(ccdf_emp)

                    x_tail = x_emp[x_emp >= xmin]
                    ccdf_tail = (x_tail / xmin) ** (1 - alpha)

                    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                    ax.plot(
                        x_emp,
                        ccdf_emp,
                        label=f"Empirical ({mode_label})",
                        color="black",
                        linewidth=1,
                    )
                    ax.plot(
                        x_tail,
                        ccdf_tail,
                        label=f"Power law ($\\alpha$={alpha:.2f})",
                        color="red",
                        linestyle="--",
                        linewidth=1,
                    )
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("CCDF")
                    ax.set_ylim(bottom=0)
                    ax.legend(loc="upper right", frameon=False)
                    ax.text(
                        0.97,
                        0.03,
                        f"$x_{{min}}$={xmin:.2f}\nR={R:.2f}, p={p:.4f}",
                        transform=ax.transAxes,
                        fontsize=summary_text_size,
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round",
                            facecolor="white",
                            alpha=0.7,
                            linewidth=0.5,
                        ),
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    fname_linear = (
                        f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.4f}_neighbor_{neighb}"
                        f"_{env_update_type}_ccdf_linear{mode_suffix}_{quality_label}_quality.pdf"
                    )
                    plt.savefig(savedir / fname_linear, bbox_inches="tight")
                    plt.close(fig)

            except Exception as e:
                print(f"Error fitting power-law for {key}: {e}")
                # Create a simple histogram instead
                fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                ax.hist(data, bins=20, alpha=0.7, edgecolor="black")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Frequency")
                # ax.set_title(
                #     f"{title}\n($\\lambda$={lmbda:.2f},$\\gamma_s$={gamma_s:.4f})",
                #     pad=2,
                # )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fname = (
                    f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.4f}_neighbor_{neighb}"
                    f"_{env_update_type}_hist_{quality_label}_quality.pdf"
                )
                plt.savefig(savedir / fname, bbox_inches="tight")
                plt.close(fig)

        else:
            print(f"Not enough valid data to fit power-law for {key} (n = {len(data)})")
            # Still create a simple plot if we have some data
            if len(data) > 0:
                fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                ax.hist(data, bins=min(10, len(data)), alpha=0.7, edgecolor="black")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Frequency")
                # ax.set_title(
                #     f"{title}\n($\\lambda$={lmbda:.2f},$\\gamma_s$={gamma_s:.4f})",
                #     pad=2,
                # )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fname = (
                    f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.4f}_neighbor_{neighb}"
                    f"_{env_update_type}_limited_{quality_label}_quality.pdf"
                )
                plt.savefig(savedir / fname, bbox_inches="tight")
                plt.close(fig)

    # Representative spatial correlation curve for this parameter combination
    profile_args = (
        seed,
        seed,
        lmbda,
        gamma_s,
        neighb,
        severity,
        radius,
        b2,
        env_update_type,
        memory_count,
        corr_window,
        acf_max_lag,
        True,
    )
    try:
        profile = run_model(profile_args)
        corr_distances = profile[-2]
        corr_values = profile[-1]
        corr_length = profile[-3]

        fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
        positive = np.isfinite(corr_values) & (corr_values > 0)
        if np.count_nonzero(positive) >= 2:
            ax.plot(corr_distances[positive], corr_values[positive], color="black")
            ax.set_yscale("log")
        else:
            ax.plot(corr_distances, corr_values, color="black")

        ax.axhline(np.exp(-1), color="gray", linestyle="--", linewidth=0.8)
        if np.isfinite(corr_length):
            ax.axvline(corr_length, color="red", linestyle="--", linewidth=1)
            ax.text(
                0.97,
                0.95,
                f"$\\xi$={corr_length:.2f}",
                transform=ax.transAxes,
                fontsize=summary_text_size,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5
                ),
            )

        ax.set_xlabel("Distance r")
        ax.set_ylabel("C(r)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        corr_fname = (
            f"spatial_corr_lambda{lmbda:.2f}_gamma{gamma_s:.4f}_neighbor_{neighb}"
            f"_{env_update_type}_{quality_label}_quality.pdf"
        )
        plt.savefig(savedir / corr_fname, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting spatial correlation for lambda={lmbda:.2f}: {e}")

    return results


if __name__ == "__main__":
    FIGURES_DIR = Path("results/figures")
    SEVERITY = "adaptive"
    ENV_UPDATE_TYPE = "piecewise"
    MEMORY_SIZE = 10
    NUM_AGENTS = 900
    PARAMS = [
        # (neigborhood prediction option, lambda, gamma_s)
        (None, 4.0, 0.0049),
        # ("linear", 4.0, 0.0049),
        (None, 4.0, 0.0045),
    ]

    QUICK_REPEATS = 100
    FULL_REPEATS = 50000

    configure_mpl()

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--corr-window", type=int, default=1)
    parser.add_argument("--acf-max-lag", type=int, default=200)
    parser.add_argument("--lambda-value", type=float)
    parser.add_argument("--gamma-s", type=float)
    parser.add_argument(
        "--neighborhood-prediction",
        choices=["none", "linear"],
        default="none",
    )
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--memory-count", type=int, default=MEMORY_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-update-type",
        choices=["linear", "piecewise"],
        default=ENV_UPDATE_TYPE,
    )
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS

    if args.repeats is not None:
        if args.repeats < 1:
            parser.error("--repeats must be >= 1")
        repeats = args.repeats

    if args.acf_max_lag < 1:
        parser.error("--acf-max-lag must be >= 1")

    if (args.lambda_value is None) != (args.gamma_s is None):
        parser.error("--lambda-value and --gamma-s must be provided together")

    if args.lambda_value is not None:
        neighb_predict_opt = (
            None
            if args.neighborhood_prediction == "none"
            else args.neighborhood_prediction
        )
        params = [(neighb_predict_opt, args.lambda_value, args.gamma_s)]
    else:
        if args.neighborhood_prediction != "none":
            parser.error(
                "--neighborhood-prediction requires --lambda-value and --gamma-s"
            )
        params = PARAMS

    b2 = np.full(NUM_AGENTS, 1.0) if args.const_peer_pressure else None

    for neighb_predict_opt, λ, γ in params:
        plot_distributions_for_param_combo(
            lmbda=λ,
            gamma_s=γ,
            neighb=neighb_predict_opt,
            severity=SEVERITY,
            b2=b2,
            env_update_type=args.env_update_type,
            repeats=repeats,
            memory_count=args.memory_count,
            seed=args.seed,
            savedir=FIGURES_DIR,
            quality_label=quality_label,
            corr_window=args.corr_window,
            acf_max_lag=args.acf_max_lag,
        )

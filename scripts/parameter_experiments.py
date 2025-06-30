import concurrent.futures
import itertools
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from abm_project.batch_run_tools import (
    analyze_environment_clusters_periodic,
    get_dominant_frequency_and_power,
)
from abm_project.mean_field import solve
from abm_project.plotting import get_file_basename, save_and_plot_heatmap
from abm_project.utils import linear_update, piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def plot_environment_for_varying_rationality(
    savedir: Path | None = None,
    gamma_s: float = 0.008,
    memory_count: int = 10,
    neighb_prediction_option=None,
    severity_benefit_option=None,
    radius_option="single",
    b_2=None,
    env_update_fn_type="linear",
    repeats: int = 30,
    rationality=None,
    num_steps: int = 1000,
):
    if rationality is None:
        rationality = np.array([0.5, 1.0, 2.0, 3.0])
    savedir = savedir or Path(".")

    # num_agents = 900
    # width = 30
    # height = 30

    # Prepare tasks for parallel execution
    tasks = []
    for i, lmbda in enumerate(rationality):
        for r in range(repeats):
            tasks.append(
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
                    num_steps,
                )
            )

    # Results containers
    env_dict = {lmbda: [] for lmbda in rationality}
    act_dict = {lmbda: [] for lmbda in rationality}
    supp_dict = {lmbda: [] for lmbda in rationality}
    press_dict = {lmbda: [] for lmbda in rationality}

    # Run in parallel using the new run_model_with_timeseries function
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (
            r,
            i,
            env_ts,
            action_ts,
            support_ts,
            pressure_ts,
        ) in executor.map(run_model_with_timeseries, tasks):
            lmbda = rationality[i]
            print(f"Completed run {r + 1} for rationality {lmbda:.2f}")
            # Store the actual time series data
            env_dict[lmbda].append(env_ts)
            act_dict[lmbda].append(action_ts)
            supp_dict[lmbda].append(support_ts)
            press_dict[lmbda].append(pressure_ts)
    print("All runs completed.")
    t = np.arange(num_steps + 1)

    # Add ODE solution for comparison
    t_ode, (n, s, sp, a, _) = solve(
        b=0.5,
        c=0.5,
        recovery=1,
        pollution=1,
        alpha=1,
        beta=1,
        n_update_rate=0.01,
        s_update_rate=gamma_s,
        n0=0.5,
        m0=0.001,
        num_steps=num_steps,
    )
    print("ODE solution computed.")

    # Plot each variable in a separate figure
    plot_info = [
        (
            "Mean environment",
            env_dict,
            n,
            "system_time_series_env_varying_rationality.png",
            (0, 1),
        ),
        (
            "Mean support",
            supp_dict,
            s,
            "system_time_series_support_varying_rationality.png",
            (0, 4),
        ),
        (
            "Mean social pressure",
            press_dict,
            sp,
            "system_time_series_pressure_varying_rationality.png",
            (0, 4),
        ),
        (
            "Mean action",
            act_dict,
            a,
            "system_time_series_action_varying_rationality.png",
            (-1, 1),
        ),
    ]
    for ylabel, data_dict, ode_data, fname, ylim in plot_info:
        fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)

        for lmbda in rationality:
            arr = np.stack(data_dict[lmbda])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1)
            ci = 1.97 * std / np.sqrt(repeats)
            ax.plot(t, mean, label=f"$\\lambda = {lmbda:.2f}$")
            ax.fill_between(t, mean - ci, mean + ci, alpha=0.3)

        ode_data = ode_data
        # ODE
        # ax.plot(t_ode,
        # ode_data,
        # linestyle="dashed",
        # linewidth=1,
        # color="black",
        # label="ODE $\lambda = 1.0$")

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time step")
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7, loc="best", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(savedir / fname, dpi=300, bbox_inches="tight")
        plt.show()
    print("Plotting complete.")


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


def run_model_with_timeseries(args):
    """Modified version of run_model that returns full time series data for plotting."""
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
        num_steps,
    ) = args
    num_agents = 900
    width = 30
    height = 30

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

    env_timeseries = model.environment[: model.time + 1].mean(axis=1)
    action_timeseries = model.action[: model.time + 1].mean(axis=1)
    support_timeseries = model.s[: model.time + 1].mean(axis=1)

    mean_local_action = (model.adj @ model.action[: model.time + 1].T).T
    if hasattr(model, "b") and len(model.b) > 1:
        pressure_timeseries = (
            model.b[1] * (model.action[: model.time + 1] - mean_local_action) ** 2
        ).mean(axis=1)
    else:
        pressure_timeseries = (
            (model.action[: model.time + 1] - mean_local_action) ** 2
        ).mean(axis=1)

    return (
        r,
        i,
        env_timeseries,
        action_timeseries,
        support_timeseries,
        pressure_timeseries,
    )


def plot_steady_state_environment_for_varying_rationality(
    savedir: Path | None = None,
    gamma_s: float = 0.01,
    memory_count_single: int = 10,
    neighb_prediction_option=None,
    severity_benefit_option=None,
    radius_option="single",
    b_2=None,
    env_update_fn_type="linear",
    repeats: int = 50,
    min_rationality: float = 0.0,
    max_rationality: float = 5.0,
):
    savedir = savedir or Path(".")
    savedir.mkdir(parents=True, exist_ok=True)

    rationality = np.linspace(min_rationality, max_rationality, 25)

    # Prepare arrays for all outputs
    results_env = np.empty((repeats, len(rationality)))
    results_num_clusters = np.empty((repeats, len(rationality)))
    results_mean_cluster_size = np.empty((repeats, len(rationality)))
    results_max_cluster_size = np.empty((repeats, len(rationality)))
    results_peak_freq = np.empty((repeats, len(rationality)))
    results_power = np.empty((repeats, len(rationality)))

    tasks = [
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
            memory_count_single,
        )
        for r in range(repeats)
        for i, lmbda in enumerate(rationality)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (
            r,
            i,
            num_clusters,
            mean_cluster_size,
            max_cluster_size,
            mean_env,
            peak_freq,
            power,
        ) in executor.map(run_model, tasks):
            print(f"Completed run {r + 1} for rationality {rationality[i]:.2f}")
            results_env[r, i] = mean_env
            results_num_clusters[r, i] = num_clusters
            results_mean_cluster_size[r, i] = mean_cluster_size
            results_max_cluster_size[r, i] = max_cluster_size
            results_peak_freq[r, i] = peak_freq
            results_power[r, i] = power

    # Helper for plotting (optimized for two-column report)
    def plot_with_ci(data, ylabel, fname, ylim=None, annotate_steepest=False):
        fig, ax = plt.subplots(
            figsize=(3.3, 2.2), constrained_layout=True
        )  # ~8.5cm x 5.5cm
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)
        ci = 1.97 * std / np.sqrt(repeats)
        ax.plot(rationality, mean, label="Mean", linewidth=1)
        ax.fill_between(rationality, mean - ci, mean + ci, alpha=0.3, linewidth=0)
        if annotate_steepest:
            dmean = np.gradient(mean, rationality)
            idx_steepest = np.argmax(np.abs(dmean))
            ax.plot(
                rationality[idx_steepest], mean[idx_steepest], "ro", label="Steepest"
            )
            ax.annotate(
                f"Steepest\n($\\lambda$={rationality[idx_steepest]:.2f})",
                xy=(rationality[idx_steepest], mean[idx_steepest]),
                xytext=(rationality[idx_steepest] + 0.2, mean[idx_steepest]),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.7),
                color="red",
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_xlabel(r"Rationality ($\lambda$)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(min_rationality, max_rationality)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=6, loc="best", frameon=False)
        # Use smaller ticks and labels for two-column
        ax.tick_params(axis="both", which="major", labelsize=7)
        fig.savefig(savedir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Plot all outputs
    plot_with_ci(
        results_env,
        r"Equilibrium mean environment ($\overline{n}^*$)",
        "equilibrium_env_vs_rationality.png",
        ylim=(0, 1),
        annotate_steepest=True,
    )
    plot_with_ci(
        results_num_clusters,
        "Number of clusters",
        "num_clusters_vs_rationality.png",
    )
    plot_with_ci(
        results_mean_cluster_size,
        "Mean cluster size",
        "mean_cluster_size_vs_rationality.png",
    )
    plot_with_ci(
        results_max_cluster_size,
        "Max cluster size",
        "max_cluster_size_vs_rationality.png",
    )
    plot_with_ci(
        results_peak_freq,
        "Dominant frequency of environment",
        "peak_freq_vs_rationality.png",
    )
    plot_with_ci(
        results_power,
        "Power at dominant frequency",
        "power_vs_rationality.png",
    )


def plot_steady_state_environment_for_varying_gamma_s(
    savedir: Path | None = None,
    rationality: float = 2.0,
    memory_count_single: int = 10,
    neighb_prediction_option=None,
    severity_benefit_option=None,
    radius_option="single",
    b_2=None,
    env_update_fn_type="linear",
    repeats: int = 50,
    min_gamma_s: float = 0.001,
    max_gamma_s: float = 0.02,
):
    savedir = savedir or Path(".")
    savedir.mkdir(parents=True, exist_ok=True)

    gamma_s_values = np.linspace(min_gamma_s, max_gamma_s, 25)

    # Prepare arrays for all outputs
    results_env = np.empty((repeats, len(gamma_s_values)))
    results_num_clusters = np.empty((repeats, len(gamma_s_values)))
    results_mean_cluster_size = np.empty((repeats, len(gamma_s_values)))
    results_max_cluster_size = np.empty((repeats, len(gamma_s_values)))
    results_peak_freq = np.empty((repeats, len(gamma_s_values)))
    results_power = np.empty((repeats, len(gamma_s_values)))

    tasks = [
        (
            r,
            i,
            rationality,
            gamma_s,
            neighb_prediction_option,
            severity_benefit_option,
            radius_option,
            b_2,
            env_update_fn_type,
            memory_count_single,
        )
        for r in range(repeats)
        for i, gamma_s in enumerate(gamma_s_values)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (
            r,
            i,
            num_clusters,
            mean_cluster_size,
            max_cluster_size,
            mean_env,
            peak_freq,
            power,
        ) in executor.map(run_model, tasks):
            print(f"Completed run {r + 1} for gamma_s {gamma_s_values[i]:.4f}")
            results_env[r, i] = mean_env
            results_num_clusters[r, i] = num_clusters
            results_mean_cluster_size[r, i] = mean_cluster_size
            results_max_cluster_size[r, i] = max_cluster_size
            results_peak_freq[r, i] = peak_freq
            results_power[r, i] = power

    # Helper for plotting
    def plot_with_ci(data, ylabel, fname, ylim=None, annotate_steepest=False):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)
        ci = 1.97 * std / np.sqrt(repeats)
        ax.plot(gamma_s_values, mean, label="Mean")
        ax.fill_between(gamma_s_values, mean - ci, mean + ci, alpha=0.3)
        if annotate_steepest:
            dmean = np.gradient(mean, gamma_s_values)
            idx_steepest = np.argmax(np.abs(dmean))
            ax.plot(
                gamma_s_values[idx_steepest],
                mean[idx_steepest],
                "ro",
                label="Steepest point",
            )
            ax.annotate(
                f"Steepest\n($\\gamma_s$={gamma_s_values[idx_steepest]:.4f})",
                xy=(gamma_s_values[idx_steepest], mean[idx_steepest]),
                xytext=(gamma_s_values[idx_steepest] + 0.002, mean[idx_steepest]),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red",
                fontsize=10,
                ha="left",
            )
        ax.set_xlabel(r"Support update rate ($\gamma_s$)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(min_gamma_s, max_gamma_s)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        fig.savefig(savedir / fname, dpi=300, bbox_inches="tight")
        plt.show()

    # Plot all outputs
    plot_with_ci(
        results_env,
        r"Equilibrium mean environment ($\overline{n}^*$)",
        "equilibrium_env_vs_gamma_s.png",
        ylim=(0, 1),
        annotate_steepest=True,
    )
    plot_with_ci(
        results_num_clusters,
        "Number of clusters",
        "num_clusters_vs_gamma_s.png",
        annotate_steepest=True,
    )
    plot_with_ci(
        results_mean_cluster_size,
        "Mean cluster size",
        "mean_cluster_size_vs_gamma_s.png",
    )
    plot_with_ci(
        results_max_cluster_size,
        "Max cluster size",
        "max_cluster_size_vs_gamma_s.png",
    )
    plot_with_ci(
        results_peak_freq,
        "Dominant frequency of environment",
        "peak_freq_vs_gamma_s.png",
    )
    plot_with_ci(
        results_power,
        "Power at dominant frequency",
        "power_vs_gamma_s.png",
    )


def run_and_animate_vectorised_model(
    results_dir: Path,
    num_agents: int = 900,
    width: int = 30,
    height: int = 30,
    num_steps: int = 1000,
    memory_count: int = 10,
    lmbda: float = 4.0,
    gamma_s: float = 0.008,
    neighb_prediction_option=None,
    severity_benefit_option=None,
    radius_option="single",
    b_2=None,
    env_update_fn_type="linear",
):
    results_dir.mkdir(exist_ok=True)

    if env_update_fn_type == "linear":
        env_update_fn = linear_update(0.01)
    elif env_update_fn_type == "piecewise":
        env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    else:
        raise ValueError(f"Unknown env_update_fn_type: {env_update_fn_type}")

    start = time.time()
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
    end = time.time()

    print(f"Model state after {num_steps} steps ({end - start:.2f}s).")
    print("======================================")
    print("Environment:")
    print(f"Mean: {model.environment[num_steps].mean():.2f}")
    print(f"Min: {model.environment[num_steps].min():.2f}")
    print(f"Max: {model.environment[num_steps].max():.2f}")
    print()
    print("Support for mitigation:")
    print(f"Mean: {model.s[num_steps].mean():.2f}")
    print(f"Min: {model.s[num_steps].min():.2f}")
    print(f"Max: {model.s[num_steps].max():.2f}")

    # Take every 5th frame for environment status animation
    env_status_history = model.environment[: model.time + 1]
    env_status_history = env_status_history.reshape((-1, model.height, model.width))
    env_status_history = env_status_history[::5]
    num_env_frames = env_status_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(env_status_history[0], cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
    ax.set_title("Agent Environment Status Over Time")

    def update_env(frame):
        im.set_array(env_status_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update_env, frames=num_env_frames, blit=True, interval=100, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Environment Status")
    plt.tight_layout()
    ani.save(results_dir / "example_env.gif", dpi=150)
    ani.save(results_dir / "example_env.mp4", dpi=150, writer="ffmpeg")

    # Take every 5th frame for agent actions animation
    agent_action_history = model.action[: model.time + 1]
    agent_action_history = agent_action_history.reshape((-1, model.height, model.width))
    agent_action_history = agent_action_history[::5]
    num_action_frames = agent_action_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(
        agent_action_history[0], cmap=ListedColormap(["red", "green"]), origin="lower"
    )
    ax.set_title("Agent Actions Over Time")

    def update_action(frame):
        im.set_array(agent_action_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update_action,
        frames=num_action_frames,
        blit=True,
        interval=100,
        repeat=False,
    )

    plt.colorbar(im, ax=ax, label="Agent Action (-1 or 1)")
    plt.tight_layout()
    ani.save(results_dir / "example_actions.gif", dpi=150)
    ani.save(results_dir / "example_actions.mp4", dpi=150, writer="ffmpeg")


def plot_distributions_for_param_combo(
    lmbda: float,
    gamma_s: float,
    neighb: str = "linear",
    severity: str = "adaptive",
    radius: str = "all",
    b2: np.ndarray = None,
    env_update_type: str = "linear",
    repeats: int = 100,
    memory_count: int = 10,
    seed: int = 42,
    savedir: Path = Path("plots"),
):
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

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
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

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
                    f"{title}\n($\\lambda$={lmbda:.2f}, $\\gamma_s$={{gamma_s:.3f}})",
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
                    savedir / f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}"
                    "_{env_update_type}_ccdf.png"
                )
                plt.tight_layout(pad=0.2)
                plt.savefig(fname, dpi=300)
                plt.show()

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
                    savedir / f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}"
                    "_{env_update_type}_hist.png"
                )
                plt.tight_layout(pad=0.2)
                plt.savefig(fname, dpi=300)
                plt.show()

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
                    savedir / f"{key}_lambda{lmbda:.2f}_gamma{gamma_s:.3f}_"
                    f"{env_update_type}_limited.png"
                )
                plt.tight_layout(pad=0.2)
                plt.savefig(fname, dpi=300)
                plt.show()

    plt.rcdefaults()
    return results


def compute_and_save_heatmap(
    repeats,
    rationality_values,
    gamma_s_values,
    neighb,
    severity,
    radius,
    b2,
    env_update_type,
    savedir,
    memory_count=10,
):
    dir_name = (
        f"neighb-{neighb}_sev-{severity}_rad-{radius}_b2-"
        f"{'1' if b2 is not None else 'None'}_env-{env_update_type}"
    )
    data_dir = savedir / dir_name
    data_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [
        "clusters",
        "mean_cluster_size",
        "max_cluster_size",
        "mean_env_status",
        "peak_freq",
        "power",
    ]
    base_paths = {
        suffix: data_dir
        / f"data_{
            get_file_basename(suffix, neighb, severity, radius, b2, env_update_type)
        }.npy"
        for suffix in suffixes
    }

    if all(path.exists() for path in base_paths.values()):
        print(
            f"Skipping: {neighb}, {severity}, {radius}, "
            f"b2={'1' if b2 is not None else 'None'}, env={env_update_type}"
        )
        data_arrays = {suffix: np.load(path) for suffix, path in base_paths.items()}
    else:
        print(
            f"Computing: {neighb}, {severity}, {radius}, "
            f"b2={'1' if b2 is not None else 'None'}, env={env_update_type}"
        )
        cluster_counts = np.zeros((len(gamma_s_values), len(rationality_values)))
        mean_cluster_sizes = np.zeros_like(cluster_counts)
        max_cluster_sizes = np.zeros_like(cluster_counts)
        mean_env_status = np.zeros_like(cluster_counts)
        peak_freqs = np.zeros_like(cluster_counts)
        power_data = np.zeros_like(cluster_counts)

        tasks = [
            (
                r,
                i * len(rationality_values) + j,
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
            for i, gamma_s in enumerate(gamma_s_values)
            for j, lmbda in enumerate(rationality_values)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            for result in executor.map(run_model, tasks):
                (
                    r,
                    flat_index,
                    num_clusters,
                    mean_size,
                    max_size,
                    mean_env,
                    peak_freq,
                    power,
                ) = result
                i = flat_index // len(rationality_values)
                j = flat_index % len(rationality_values)
                cluster_counts[i, j] += num_clusters / repeats
                mean_cluster_sizes[i, j] += mean_size / repeats
                max_cluster_sizes[i, j] += max_size / repeats
                mean_env_status[i, j] += mean_env / repeats
                peak_freqs[i, j] += peak_freq / repeats
                power_data[i, j] += power / repeats
                r = r

        data_arrays = {
            "clusters": cluster_counts,
            "mean_cluster_size": mean_cluster_sizes,
            "max_cluster_size": max_cluster_sizes,
            "mean_env_status": mean_env_status,
            "peak_freq": peak_freqs,
            "power": power_data,
        }

        for suffix, data in data_arrays.items():
            np.save(base_paths[suffix], data)

    return data_arrays


def plot_heatmap_env_vs_rationality_and_gamma_s(
    savedir: Path | None = None, memory_count: int = 10
):
    savedir = savedir or Path(".")
    savedir.mkdir(parents=True, exist_ok=True)

    repeats = 25
    rationality_values = np.linspace(0, 6.0, 20)
    gamma_s_values = np.linspace(0.001, 0.02, 20)

    neighb_opts = ["linear", None]
    severity_opts = ["adaptive", None]
    radius_opts = ["single", "all"]
    b2_opts = [None, np.full(900, 1.0)]
    env_update_opts = ["linear", "piecewise"]

    combinations = list(
        itertools.product(
            neighb_opts, severity_opts, radius_opts, b2_opts, env_update_opts
        )
    )

    for neighb, severity, radius, b2, env_update_type in combinations:
        data_arrays = compute_and_save_heatmap(
            repeats,
            rationality_values,
            gamma_s_values,
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            memory_count=memory_count,
        )
        save_and_plot_heatmap(
            data_arrays["clusters"],
            "Number of Clusters",
            "clusters",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )
        save_and_plot_heatmap(
            data_arrays["mean_cluster_size"],
            "Mean Cluster Size",
            "mean_cluster_size",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )
        save_and_plot_heatmap(
            data_arrays["max_cluster_size"],
            "Max Cluster Size",
            "max_cluster_size",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )
        save_and_plot_heatmap(
            data_arrays["mean_env_status"],
            "Mean Environmental Status",
            "mean_env_status",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )
        save_and_plot_heatmap(
            data_arrays["peak_freq"],
            "Dominant Frequency of Environment",
            "peak_freq",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )
        save_and_plot_heatmap(
            data_arrays["power"],
            "Power at Dominant Frequency",
            "power",
            neighb,
            severity,
            radius,
            b2,
            env_update_type,
            savedir,
            rationality_values,
            gamma_s_values,
        )


if __name__ == "__main__":
    results_dir = Path("plots")
    results_dir.mkdir(exist_ok=True)

    plot_heatmap_env_vs_rationality_and_gamma_s(savedir=results_dir, memory_count=10)

    lambda_single = 4.0
    gamma_s_single = 0.0044
    neighb_single = "linear"
    severity_single = "adaptive"  # "adaptive"
    radius_single = "single"  # "single" or "all"
    b2_single = None  # np.full(900, 1.0)
    env_update_type_single = "piecewise"  # "linear" or "piecewise"
    memory_count_single = 10

    plot_steady_state_environment_for_varying_rationality(
        savedir=results_dir,
        gamma_s=gamma_s_single,
        memory_count_single=memory_count_single,
        neighb_prediction_option=neighb_single,
        severity_benefit_option=severity_single,
        radius_option=radius_single,
        b_2=b2_single,
        env_update_fn_type=env_update_type_single,
        repeats=10,
        min_rationality=0.0,
        max_rationality=2.0,
    )

    plot_steady_state_environment_for_varying_gamma_s(
        savedir=results_dir,
        rationality=lambda_single,
        memory_count_single=memory_count_single,
        neighb_prediction_option=neighb_single,
        severity_benefit_option=severity_single,
        radius_option=radius_single,
        b_2=b2_single,
        env_update_fn_type=env_update_type_single,
        repeats=50,
        min_gamma_s=0.001,
        max_gamma_s=0.01,
    )

    plot_environment_for_varying_rationality(
        savedir=results_dir,
        gamma_s=gamma_s_single,
        memory_count=memory_count_single,
        neighb_prediction_option=neighb_single,
        severity_benefit_option=severity_single,
        radius_option=radius_single,
        b_2=b2_single,
        env_update_fn_type=env_update_type_single,
        repeats=10,
        rationality=np.array([1.0, 2.0, 4.0, 6.0]),
        num_steps=2000,
    )

    plot_distributions_for_param_combo(
        lmbda=lambda_single,
        gamma_s=gamma_s_single,
        neighb=neighb_single,
        severity=severity_single,
        env_update_type=env_update_type_single,
        repeats=500,
        memory_count=memory_count_single,
        savedir=results_dir,
    )

    run_and_animate_vectorised_model(
        results_dir=results_dir,
        num_agents=900,
        width=30,
        height=30,
        num_steps=2000,
        memory_count=memory_count_single,
        lmbda=lambda_single,
        gamma_s=gamma_s_single,
        neighb_prediction_option=neighb_single,
        severity_benefit_option=severity_single,
        radius_option=radius_single,
        b_2=b2_single,
        env_update_fn_type=env_update_type_single,
    )

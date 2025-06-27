import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ruptures import Pelt
from tqdm import tqdm

from abm_project.cluster_analysis import (
    cluster_time_series,
    correlation_length_time_series,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel

# from abm_project.utils_phase_plt import plot_phase_portrait_abm


# TODO: Update vectorized model to use the new parameters
def test_cluster_across_memory(option):
    # Parameters for the simulation
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 500
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None
    rationality = 1.8
    simmer_time = 1
    neighb_prediction_option = "linear"
    # severity_benefit_option = None
    memory_values = [x for x in range(5, 101, 5)]

    # Analysis parameters
    replicates = 20
    critical_times = {m: [] for m in memory_values}
    cluster_n = {n: [] for n in memory_values}
    largest_cluster = {n: [] for n in memory_values}

    for mem_count in memory_values:
        for rep in tqdm(
            range(replicates), desc=f"Memory Count: {mem_count}, Replicate: "
        ):
            print(f"Running replicate {rep + 1} for memory count {mem_count}...")
            results_dir = Path("cluster_analysis_results")
            results_dir.mkdir(exist_ok=True)

            memory_count = mem_count

            # start = time.time()
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=rationality,
                simmer_time=simmer_time,
                neighb_prediction_option=neighb_prediction_option,
                severity_benefit_option=None,
                max_storage=num_steps,
            )
            model.run(num_steps)
            # end = time.time()

            Nc, C1 = cluster_time_series(model=model, option=option)

            # 3) Detect transition time t_c via change‐point on Nc(t)
            algo = Pelt(model="rbf", min_size=5).fit(Nc)
            bkpts = algo.predict(pen=3)  # list of breakpoints
            critical_times[mem_count].append(bkpts[0])  # first change point

    # 4) Summarize: average t_c vs memory
    for m in memory_values:
        print(f"Memory {m}:\nmean t_c = {np.mean(critical_times[m]):.1f}")
        print(f"std = {np.std(critical_times[m]):.1f}")

    return critical_times, cluster_n, largest_cluster, memory_values


# TODO: Fix it. Doesn't work yet
def plot_cluster_across_memory(critical_times, option, savedir):
    savedir = savedir or Path(".")

    memory_values = list(critical_times.keys())

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        memory_values,
        [np.mean(critical_times[m]) for m in memory_values],
        yerr=[np.std(critical_times[m]) for m in memory_values],
        fmt="o-",
        capsize=5,
    )
    plt.xlabel("Memory Count")
    plt.ylabel("Critical Time (t_c)")
    plt.title("Critical Time vs Memory Count")
    plt.grid()

    if savedir:
        plt.savefig(
            savedir / f"cluster_across_memory_{option}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved as 'cluster_across_memory_{option}.png'")
    # plt.show()
    plt.close()


def plot_ncluster_given_memory(model, option, num_steps, savedir):
    savedir = savedir or Path(".")

    # Nc, C1 = cluster_time_series(model=model, option=option)
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
        # print(
        #     f"Plot saved as 'mem{model.memory_count}_{option}_
        # {model.severity_benefit_option}_{model.rationality}_{model.width}_
        # {model.height}_{num_steps}.png'"
        # )
    # plt.show()
    plt.close()


def plot_ncluster_against_memory_rationality(
    models, option, memory_range, rat_range, savedir
):
    #  Create a heatmap of memory vs rationality filled with number of clusters

    savedir = savedir or Path(".")

    # Plot equilibrium env state (environment state at last timestep)
    # initialize 2d array to store env equilibrium state
    eq_env_state = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
            # Find the model with the current memory and rationality
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model:
                if option == "action":
                    history = model.action[: model.time + 1]
                elif option == "environment":
                    history = model.environment[: model.time + 1]
                history = history.reshape((-1, model.height, model.width))
                lattice = history[history.shape[0] - 1]
                eq_env_state[i, j] = np.mean(lattice)

    # Save equilibrium state to a npz file
    filepath = Path(savedir) / "eq_env_state.npz"
    np.savez(
        filepath,
        eq_env_state=eq_env_state,
        memory_range=memory_range,
        rat_range=rat_range,
    )

    # Initialize a 2D array to store the number of clusters
    # n_clusters = np.zeros((len(memory_range), len(rat_range)))
    # for i, mem in enumerate(memory_range):
    #     for j, rat in enumerate(rat_range):
    #         # Find the model with the current memory and rationality
    #         model = next((m for m in models if m.memory_count == mem and
    # m.rationality == rat), None)
    #         if model:
    #            nc, _ = cluster_time_series(model=model, option=option)
    #             n_clusters[i, j] = np.mean(nc)  # Average number of clusters over time

    plt.figure(figsize=(12, 8))

    # plt.imshow(n_clusters, aspect='auto', cmap='viridis', origin='lower',
    #            extent=[rat_range[0], rat_range[-1], memory_range[0],
    # memory_range[-1]])

    # plot equilibrium env state (environment state at last timestep)
    plt.imshow(
        eq_env_state,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[rat_range[0], rat_range[-1], memory_range[0], memory_range[-1]],
    )

    plt.colorbar(label="Equilibrium Environment")
    plt.xlabel("Rationality (rat)", fontsize=15)
    plt.ylabel("Memory Count (mem_count)", fontsize=15)
    plt.title("Steady-State Environment as Function of rat and mem_count", fontsize=18)
    plt.xticks(rat_range)
    plt.xticks(fontsize=15)
    plt.xticks(rotation=90)
    plt.yticks(memory_range)
    plt.yticks(fontsize=15)

    if savedir:
        plt.savefig(
            savedir
            / f"n_clusters_vs_memory({memory_count_range[0]},{memory_count_range[-1]})_\
            rationality({rat_range[0]},{rat_range[-1]})_gs{models[0].gamma_s}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved as 'cluster_across_memory_{option}.png'")
    # plt.show()
    plt.close()


# plot heatmap from eq_env_state.npz in a new function below
def plot_heatmap_from_npz(npz_file, savedir=None):
    """Plot heatmap from a npz file containing equilibrium environment state."""
    savedir = savedir or Path(".")

    data = np.load(npz_file)
    eq_env_state = data["eq_env_state"]
    memory_range = data["memory_range"]
    rat_range = data["rat_range"]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        eq_env_state,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=[rat_range[0], rat_range[-1], memory_range[0], memory_range[-1]],
    )

    cb = plt.colorbar(label="Equilibrium Environment")
    cb.set_label(label="Equilibrium Environment", fontsize=15)
    plt.xlabel(r"Rationality ($\lambda$)", fontsize=15)
    plt.ylabel("Memory Count (mem_count)", fontsize=15)
    # plt.title('Steady-State Environment as Function of rat
    # and mem_count', fontsize=18)
    plt.xticks(rat_range, fontsize=15)
    plt.locator_params(axis="x", nbins=15)
    # make xticks vertical
    plt.xticks(rotation=90)
    plt.yticks(memory_range, fontsize=15)
    plt.locator_params(axis="y", nbins=15)
    plt.tight_layout()

    if savedir:
        plt.savefig(
            savedir
            / f"n_clusters_vs_memory({memory_count_range[0]},{memory_count_range[-1]})\
            _rationality({rat_range[0]},{rat_range[-1]}).png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Heatmap saved as 'heatmap_eq_env_state.png'")

    # plt.show()
    plt.close()


def plot_ncluster_across_memory(cluster_n, option, savedir):
    savedir = savedir or Path(".")

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


# def main():
# Set model parameters
def run_model(
    num_agents=2500,
    width=50,
    height=50,
    num_steps=1000,
    memory_count=40,
    rationality=0.2,
    env_update_fn=None,
    rng=None,
    neighb_prediction_option="linear",
    severity_benefit_option="adaptive",
    gamma_s=0.002,
):
    """Run the agent-based model with specified parameters."""

    start = time.time()

    # Create the model instance with the specified parameters
    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality,
        neighb_prediction_option=neighb_prediction_option,
        severity_benefit_option=severity_benefit_option,
        max_storage=num_steps,
        gamma_s=gamma_s,
    )

    model.run(num_steps)
    end = time.time()

    print(f"Simulation completed in {end - start:.2f} seconds.")

    return model, num_steps


def run_model_multiple_runs(
    num_agents=2500,
    width=50,
    height=50,
    num_steps=1000,
    memory_count=40,
    rationality=0.1,
    env_update_fn=None,
    rng=None,
    neighb_prediction_option="linear",
    severity_benefit_option="adaptive",
    gamma_s=0.01,
):
    # Run model for 50 iterations and plot the number of clusters, a trend line
    num_runs = 50
    models = []
    for i in tqdm(range(num_runs), desc="Running multiple model iterations"):
        print(f"Running model iteration {i + 1} of {num_runs}...")
        model, _ = run_model(
            num_agents=num_agents,
            width=width,
            height=height,
            num_steps=num_steps,
            memory_count=memory_count,
            rationality=rationality,
            env_update_fn=env_update_fn,
            rng=rng,
            neighb_prediction_option=neighb_prediction_option,
            severity_benefit_option=severity_benefit_option,
            gamma_s=gamma_s,
        )
        models.append(model)

    return models


def plot_ncluster_multiple_model_runs(num_steps=1000, models=None, savedir=None):
    # """Plot the average number of clusters across multiple model runs."""
    # check if models is provided
    if models is None:
        raise ValueError("No models provided for plotting.")

    savedir = savedir or Path(".")

    # Initialize lists to store the number of clusters at each time step
    num_clusters = []
    max_cluster_sizes = []
    for model in models:
        nc, c1 = cluster_time_series(model=model, option="environment")
        num_clusters.append(nc)
        max_cluster_sizes.append(c1)

    # Calculate the average number of clusters and max cluster sizes across all models
    avg_num_clusters = np.mean(num_clusters, axis=0)
    # avg_max_cluster_sizes = np.mean(max_cluster_sizes, axis=0)

    # Calculate the standard deviation for error bars
    std_num_clusters = np.std(num_clusters, axis=0)
    # std_max_cluster_sizes = np.std(max_cluster_sizes, axis=0)

    # Plot the average number of clusters
    timesteps = np.arange(len(avg_num_clusters))

    # Plot mean and shaded std
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_num_clusters, label="Mean # Clusters", color="blue")
    plt.fill_between(
        timesteps,
        avg_num_clusters - std_num_clusters,
        avg_num_clusters + std_num_clusters,
        alpha=0.3,
        label="±1 Std Dev",
        color="blue",
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


# ============================================================


#  Correlation-length scans and plotting
def _mean_final_xi(
    models: Sequence[VectorisedModel],
    *,
    num_steps: int,
    option: str = "action",
) -> tuple[float, float]:
    """
    Run each model, compute ξ(t), and return mean ± std of ξ at t = num_steps.

    Args:
        models: Iterable of *freshly initialised* VectorisedModel objects.
        num_steps: How many steps to simulate each model.
        option: "action" | "environment" — which lattice to analyse.

    Returns:
        Tuple (mean_ξ, std_ξ) across all runs.
    """
    finals = []
    for mdl in models:
        mdl.run(num_steps)
        stats = correlation_length_time_series(mdl, option=option)
        finals.append(stats["xi"][-1])  # ξ at final timestep
    finals = np.asarray(finals)
    return finals.mean(), finals.std(ddof=1)


def scan_parameter_for_xi(
    param_name: str,
    param_values: Sequence[Any],
    base_kwargs: dict[str, Any],
    *,
    num_steps: int = 1000,
    runs_per_value: int = 5,
    option: str = "action",
    loglog: bool = True,
    ax: plt.Axes | None = None,
    **plot_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep one model parameter, measure the correlation length, and plot it.

    Args:
        param_name: Keyword argument of VectorisedModel to vary
                    (e.g. "rationality", "memory_count", "gamma_s").
        param_values: Iterable of values to test.
        base_kwargs: Dict of all other VectorisedModel kwargs kept fixed.
        num_steps: Simulation length for each model.
        runs_per_value: Independent realisations per parameter point.
        option: "action" or "environment".
        ax: Existing matplotlib Axes to draw on (creates one if None).
        **plot_kwargs: Style arguments forwarded to `ax.errorbar`.

    Returns:
        means, stds: Arrays of mean ξ and std ξ for each parameter value.
    """
    means, stds = [], []

    for val in param_values:
        kwargs = base_kwargs | {param_name: val}
        models = [VectorisedModel(**kwargs) for _ in range(runs_per_value)]
        mu, sigma = _mean_final_xi(models, num_steps=num_steps, option=option)
        means.append(mu)
        stds.append(sigma)

    means = np.asarray(means)
    stds = np.asarray(stds)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if loglog:
        # Require strictly positive data for log scale
        if np.any(param_values <= 0) or np.any(means <= 0):
            raise ValueError("log-log plot requested but data contain ≤ 0 values.")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.errorbar(param_values, means, yerr=stds, marker="o", capsize=3, **plot_kwargs)
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"Correlation length $\xi$ (final timestep)")
    ax.set_title(rf"$\xi$ vs {param_name}")
    ax.grid(ls=":")
    # ax.savefig(
    #     f"cluster_analysis_results/extra/xi_vs_{param_name}.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    return means, stds


def analyse_cmax(
    model,
    *,
    num_steps: int = 1000,
    option: str = "action",
    normalise: bool = True,
    detect_perc: bool = True,
    perc_threshold: float = 0.59,
    plot_xi: bool = True,
    savedir: str | Path | None = None,
    show: bool = False,
):
    """
    Analyse and plot the largest-cluster series c_max(t).

    Args:
        model: BaseModel
            Must already have model.run(N) executed.
        option: str
            "action" (default) or "environment" - passed to cluster_time_series().
        normalise: bool
            Plot c_max as fraction of total agents (c_max / num_agents).
        detect_perc: bool
            If True, return the first timestep where c_max crosses
            `perc_threshold * num_agents`.
        perc_threshold: float
            Fraction (0-1) used for percolation detection.
        plot_xi: bool
            Also compute ξ(t) and show c_max versus ξ scatter (log-log).
        savedir: str | Path | None
            Folder to dump PNGs.  If None, nothing is saved.
        show: bool
            Whether to display the figures.

    Returns:
        dict:
            {
                "c_max": np.ndarray,            # size time-series
                "percolation_time": int | None, # first crossing or None
            }
    """
    # ------------------------------------------------ cluster sizes ----------
    _labels, nc, c_max = cluster_time_series(model, option=option)
    t = np.arange(len(c_max))
    y = c_max / model.num_agents if normalise else c_max

    fig_ts, ax_ts = plt.subplots(figsize=(6, 3.5))
    ax_ts.plot(t, y, lw=1.4)
    ax_ts.set_xlabel("Timestep")
    ax_ts.set_ylabel(
        "Largest cluster fraction" if normalise else "Largest cluster size"
    )
    ax_ts.set_title("Largest cluster vs time")
    ax_ts.grid(ls=":")

    # ------------------------------------------------ percolation time -------
    perc_time = None
    if detect_perc:
        thresh = perc_threshold * (1.0 if normalise else model.num_agents)
        hits = np.where(y >= thresh)[0]
        if hits.size:
            perc_time = int(hits[0])
            ax_ts.axvline(perc_time, color="red", ls="--", lw=1, alpha=0.8)
            ax_ts.text(
                perc_time,
                y.max() * 0.9,
                f"  percolation at t={perc_time}",
                color="red",
                va="top",
            )

    # ------------------------------------------------ scatter vs ξ ----------
    if plot_xi:
        stats = correlation_length_time_series(model, option=option)
        xi = stats["xi"]

        fig_sc, ax_sc = plt.subplots(figsize=(4, 4))
        ax_sc.scatter(c_max, xi, s=10, alpha=0.7)
        ax_sc.set_xscale("log")
        ax_sc.set_yscale("log")
        ax_sc.set_xlabel("Largest cluster size $c_{max}$")
        ax_sc.set_ylabel("Correlation length $\\xi$")
        ax_sc.set_title("$c_{max}$ vs $\\xi$")
        ax_sc.grid(ls=":", which="both")

    # ------------------------------------------------ save / show -----------
    if savedir is not None:
        savedir = Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        fig_ts.savefig(
            savedir
            / f"cmax_timeseries_randomenv_rat{model.rationality:.1f}_\
            mem{model.memory_count}.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plot_xi:
            fig_sc.savefig(
                savedir
                / f"cmax_vs_xi_randomenv_rat{model.rationality:.1f}_\
                mem{model.memory_count}.png",
                dpi=300,
                bbox_inches="tight",
            )

    if show:
        plt.show()
    else:
        plt.close("all")

    return {
        "c_max": c_max,
        "percolation_time": perc_time,
    }


# ------------------------------------------------------------
#  Example usage (executes if you run this file as a script)
# ------------------------------------------------------------
if __name__ == "__main__":
    # All the *fixed* parameters for every run
    base = dict(
        width=50,
        height=50,
        num_agents=2500,
        memory_count=20,
        rationality=0.9,
        gamma_s=0.01,
        env_update_fn=piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01),
        rng=None,
        neighb_prediction_option="linear",
        severity_benefit_option="adaptive",
    )

    # # Choose which parameter to sweep
    # betas = np.linspace(0, 2.1, 20)
    # # betas = np.logspace(-1.5, 0.3, 12)

    # scan_parameter_for_xi(
    #     "rationality",           # param to vary
    #     betas,                   # values to try
    #     base_kwargs=base,
    #     num_steps=1000,
    #     runs_per_value=10,        # 8 independent runs per β
    #     option="action",
    #     loglog=True,
    #     color="tab:blue",
    # )

    # plt.tight_layout()
    # plt.show()

    # Define range for model parameters
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000

    memory_count_range = [a for a in np.arange(2, 52, 2)]
    rationality_range = np.linspace(0, 2, 30, endpoint=True)
    # memory_count = 20
    # rationality = 0.9
    gamma_s = 0.004

    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    # env_update_fn = linear_update(rate=0.01)
    # env_update_fn = exponential_update(rate=0.01)
    # env_update_fn = sigmoid_update(rate=0.01)

    rng = None
    neighb_prediction_option = "linear"  # or "logistic"
    severity_benefit_option = "adaptive"  # or None

    models = []

    # mdl = VectorisedModel(**base)
    # for mem_count in memory_count_range:
    #     for rat_val in tqdm(rationality_range, desc=f"Memory: {mem_count},
    #  Rationality"):
    #         model = VectorisedModel(
    #                     num_agents=num_agents,
    #                     width=width,
    #                     height=height,
    #                     memory_count=mem_count,
    #                     rng=rng,
    #                     env_update_fn=env_update_fn,
    #                     rationality=rat_val,
    #                     simmer_time=1,
    #                     neighb_prediction_option=neighb_prediction_option,
    #                     severity_benefit_option=severity_benefit_option,
    #                     gamma_s=gamma_s,
    #                     max_storage=num_steps,
    #                 )
    #         model.run(num_steps)
    #         models.append(model)

    # res = analyse_cmax(
    #     mdl,
    #     num_steps = 1000,
    #     option="environment",
    #     normalise=True,
    #     savedir="cluster_analysis_results/extra/cmax_analysis",
    #     )
    # print("Percolation time:", res["percolation_time"])

    # plot_ncluster_against_memory_rationality(
    #     models=models,
    #     option="environment",
    #     memory_range=memory_count_range,
    #     rat_range=rationality_range,
    #     savedir=Path("cluster_analysis_results/extra")
    # )

    plot_heatmap_from_npz(
        npz_file=Path("cluster_analysis_results/extra/eq_env_state.npz"),
        savedir=Path("cluster_analysis_results/extra"),
    )

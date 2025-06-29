"""Script to run cluster analysis for varying parameters."""

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


# TODO: Update vectorized model to use the new parameters
def test_cluster_across_memory(model, option):
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


def eq_env_against_memory_rationality(
    models=None, memory_range=None, rat_range=None, savedir=None
):
    #  Create a heatmap of memory vs rationality filled with number of clusters
    savedir = savedir or Path(".")

    eq_env_state = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
            # Find the model with the current memory and rationality
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model:
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

    return None


def nclusters_against_memory_rationality(
    models=None, option="environment", memory_range=None, rat_range=None, savedir=None
):
    savedir = savedir or Path(".")

    # Initialize a 2D array to store the number of clusters
    n_clusters = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
            # Find the model with the current memory and rationality
            model = next(
                (m for m in models if m.memory_count == mem and m.rationality == rat),
                None,
            )
            if model:
                nc, _ = cluster_time_series(model=model, option=option)
                n_clusters[i, j] = np.mean(nc)  # Average number of clusters over time

        # Save equilibrium state to a npz file
        filepath = Path(savedir) / "n_clusters.npz"
        np.savez(
            filepath,
            n_clusters=n_clusters,
            memory_range=memory_range,
            rat_range=rat_range,
        )
    return None


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
    # ---------- cluster sizes ----------
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

    # ---------- percolation time -------
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

    # ---------- scatter vs ξ ----------
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


def main():
    # Model parameters
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000

    memory_count_range = [a for a in np.arange(2, 52, 2)]
    rationality_range = np.linspace(0, 2, 30, endpoint=True)
    # memory_count = [20]
    # rationality = [0.9]
    gamma_s = 0.004

    # Environment update parameters
    recovery = 1
    pollution = 1
    gamma = 0.01

    env_update_fn = piecewise_exponential_update(
        recovery=recovery, pollution=pollution, gamma=gamma
    )

    rng = None
    neighb_prediction_option = "linear"  # or "logistic"
    severity_benefit_option = "adaptive"  # or None

    models = []

    for memory_count in memory_count_range:
        for rationality in rationality_range:
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=rationality,
                gamma_s=gamma_s,
                neighb_prediction_option=neighb_prediction_option,
                severity_benefit_option=severity_benefit_option,
                max_storage=num_steps,
            )
            models.append(model)


if __name__ == "__main__":
    models = main()

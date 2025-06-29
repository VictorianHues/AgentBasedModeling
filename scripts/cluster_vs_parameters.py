"""Script to run cluster analysis for varying parameters."""

from pathlib import Path

import numpy as np

from abm_project.cluster_analysis import (
    cluster_time_series,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def eq_env_against_memory_rationality(
    models=None, memory_range=None, rat_range=None, savedir=None
):
    """Calculate the equilibrium environment state for varying parameters.

    Args:
        models: List of VectorisedModel instances with varying
                memory counts and rationalities.
        memory_range: List of memory counts to test.
        rat_range: List of rationality values to test.
        savedir: Directory to save the npz file. If None,
                    saves in current directory.
    """
    savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    eq_env_state = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
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
    """Calculate the number of clusters for varying memory and rationality.

    Args:
        models: List of VectorisedModel instances with varying
                memory counts and rationalities.
        option: Option indicating the type of analysis (e.g., "environment").
        memory_range: List of memory counts to test.
        rat_range: List of rationality values to test.
        savedir: Directory to save the npz file. If None,
                    saves in current directory.
    """
    savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

    n_clusters = np.zeros((len(memory_range), len(rat_range)))
    for i, mem in enumerate(memory_range):
        for j, rat in enumerate(rat_range):
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


# TODO: Sort out how to save required data to modularize
# def analyse_cmax(
#     model,
#     *,
#     num_steps: int = 1000,
#     option: str = "action",
#     normalise: bool = True,
#     detect_perc: bool = True,
#     perc_threshold: float = 0.59,
#     plot_xi: bool = True,
#     savedir: str | Path | None = None,
#     show: bool = False,
# ):
#     """
#     Analyse and plot the largest-cluster series c_max(t).

#     Args:
#         model: BaseModel
#             Must already have model.run(N) executed.
#         option: str
#             "action" (default) or "environment" - passed to cluster_time_series().
#         normalise: bool
#             Plot c_max as fraction of total agents (c_max / num_agents).
#         detect_perc: bool
#             If True, return the first timestep where c_max crosses
#             `perc_threshold * num_agents`.
#         perc_threshold: float
#             Fraction (0-1) used for percolation detection.
#         plot_xi: bool
#             Also compute ξ(t) and show c_max versus ξ scatter (log-log).
#         savedir: str | Path | None
#             Folder to dump PNGs.  If None, nothing is saved.
#         show: bool
#             Whether to display the figures.

#     Returns:
#         dict:
#             {
#                 "c_max": np.ndarray,            # size time-series
#                 "percolation_time": int | None, # first crossing or None
#             }
#     """
#     savedir = savedir or Path(savedir).mkdir(parents=True, exist_ok=True)

#     # Cluster sizes
#     _labels, nc, c_max = cluster_time_series(model, option=option)
#     t = np.arange(len(c_max))
#     y = c_max / model.num_agents if normalise else c_max

#     fig_ts, ax_ts = plt.subplots(figsize=(6, 3.5))
#     ax_ts.plot(t, y, lw=1.4)
#     ax_ts.set_xlabel("Timestep")
#     ax_ts.set_ylabel(
#         "Largest cluster fraction" if normalise else "Largest cluster size"
#     )
#     ax_ts.set_title("Largest cluster vs time")
#     ax_ts.grid(ls=":")

#     # Percolation time
#     perc_time = None
#     if detect_perc:
#         thresh = perc_threshold * (1.0 if normalise else model.num_agents)
#         hits = np.where(y >= thresh)[0]
#         if hits.size:
#             perc_time = int(hits[0])
#             ax_ts.axvline(perc_time, color="red", ls="--", lw=1, alpha=0.8)
#             ax_ts.text(
#                 perc_time,
#                 y.max() * 0.9,
#                 f"  percolation at t={perc_time}",
#                 color="red",
#                 va="top",
#             )

#     # Save data to npz file
#     if savedir is not None:
#         savedir = Path(savedir)
#         savedir.mkdir(parents=True, exist_ok=True)
#         np.savez(
#             savedir / f"cmax_timeseries_randomenv_rat{model.rationality:.1f}_\
#             mem{model.memory_count}.npz",
#             c_max=c_max,
#             t=t,
#             perc_time=perc_time,
#         )

#     return {
#         "c_max": c_max,
#         "percolation_time": perc_time,
#     }

# TODO: Need to figure out if this is still needed
# def test_cluster_across_memory(models, memory_range, option):
#     """ Test cluster analysis across different memory counts.

#     Args:
#         models: List of VectorisedModel instances with varying memory counts.
#         memory_range: List of memory counts to test.
#         option: Option indicating the type of analysis (e.g., "environment").

#     Returns:
#         critical_times: Dictionary with memory counts as keys and lists
#                         of critical times as values.
#         cluster_n: Dictionary with memory counts as keys and lists of
#                     average number of clusters as values.
#         largest_cluster: Dictionary with memory counts as keys and lists
#                             of largest cluster sizes as values.
#         memory_values: List of memory counts used in the analysis.
#     """
#     # Analysis parameters
#     replicates = 20
#     critical_times = {m: [] for m in memory_range}
#     cluster_n = {n: [] for n in memory_range}
#     largest_cluster = {n: [] for n in memory_range}

#     for i, mem in enumerate(memory_range):
#         # Find the model with the current memory
#         model = next(
#             (m for m in models if m.memory_count == mem),
#             None,
#         )
#         Nc, C1 = cluster_time_series(model=model, option=option)
#         # Store average number of clusters and largest cluster size
#         cluster_n[mem].append(np.mean(Nc))
#         largest_cluster[mem].append(np.mean(C1))

#         # Detect transition time t_c via change‐point on Nc(t)
#         algo = Pelt(model="rbf", min_size=5).fit(Nc)
#         bkpts = algo.predict(pen=3)                         # list of breakpoints
#         critical_times[memory_range].append(bkpts[0])       # first change point

#     # Summarize: average t_c vs memory
#     for m in memory_range:
#         print(f"Memory {m}:\nmean t_c = {np.mean(critical_times[m]):.1f}")
#         print(f"std = {np.std(critical_times[m]):.1f}")

#     return critical_times, cluster_n, largest_cluster


# #  Correlation-length scans and plotting
# def _mean_final_xi(
#     models: Sequence[VectorisedModel],
#     *,
#     num_steps: int,
#     option: str = "action",
# ) -> tuple[float, float]:
#     """
#     Run each model, compute ξ(t), and return mean ± std of ξ at t = num_steps.

#     Args:
#         models: Iterable of *freshly initialised* VectorisedModel objects.
#         num_steps: How many steps to simulate each model.
#         option: "action" | "environment" — which lattice to analyse.

#     Returns:
#         Tuple (mean_ξ, std_ξ) across all runs.
#     """
#     finals = []
#     for mdl in models:
#         mdl.run(num_steps)
#         stats = correlation_length_time_series(mdl, option=option)
#         finals.append(stats["xi"][-1])  # ξ at final timestep
#     finals = np.asarray(finals)
#     return finals.mean(), finals.std(ddof=1)


# def scan_parameter_for_xi(
#     param_name: str,
#     param_values: Sequence[Any],
#     base_kwargs: dict[str, Any],
#     *,
#     num_steps: int = 1000,
#     runs_per_value: int = 5,
#     option: str = "action",
#     loglog: bool = True,
#     ax: plt.Axes | None = None,
#     **plot_kwargs,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Sweep one model parameter, measure the correlation length, and plot it.

#     Args:
#         param_name: Keyword argument of VectorisedModel to vary
#                     (e.g. "rationality", "memory_count", "gamma_s").
#         param_values: Iterable of values to test.
#         base_kwargs: Dict of all other VectorisedModel kwargs kept fixed.
#         num_steps: Simulation length for each model.
#         runs_per_value: Independent realisations per parameter point.
#         option: "action" or "environment".
#         ax: Existing matplotlib Axes to draw on (creates one if None).
#         **plot_kwargs: Style arguments forwarded to `ax.errorbar`.

#     Returns:
#         means, stds: Arrays of mean ξ and std ξ for each parameter value.
#     """
#     means, stds = [], []

#     for val in param_values:
#         kwargs = base_kwargs | {param_name: val}
#         models = [VectorisedModel(**kwargs) for _ in range(runs_per_value)]
#         mu, sigma = _mean_final_xi(models, num_steps=num_steps, option=option)
#         means.append(mu)
#         stds.append(sigma)

#     means = np.asarray(means)
#     stds = np.asarray(stds)

#     return means, stds


def main():
    # Model parameters
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000

    # memory_count_range = [a for a in np.arange(2, 52, 2)]
    memory_count_range = None
    # rationality_range = np.linspace(0, 2, 30, endpoint=True)
    rationality_range = None
    memory_count = 20
    rationality = 0.9
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

    if memory_count_range and rationality_range:
        # Create models for all combinations of memory_count and rationality
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
    else:
        print(f"Using memory_count={memory_count} and rationality={rationality}")
        # Create a single model with specified memory_count and rationality
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

    return models


if __name__ == "__main__":
    models = main()
    print(f"Running cluster analysis for {len(models)} models...")

    eq_env_against_memory_rationality(
        models=models,
        memory_range=[m.memory_count for m in models],
        rat_range=[m.rationality for m in models],
        savedir=Path("data/eq_env_vs_memory_rationality"),
    )

    nclusters_against_memory_rationality(
        models=models,
        option="environment",
        memory_range=[m.memory_count for m in models],
        rat_range=[m.rationality for m in models],
        savedir=Path("data/n_clusters_vs_memory_rationality"),
    )

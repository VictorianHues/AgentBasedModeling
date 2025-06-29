import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from matplotlib.axes import Axes
from SALib.analyze import pawn, sobol
from SALib.sample import sobol as sobol_sample

from abm_project import cluster_analysis, metrics
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


# get parameter values
def problem():
    problem = {
        "num_vars": 6,
        "names": [
            "width",
            "rationality",
            "memory_count",
            "adaptation_speed",
            "radius",
            "recovery_rate",
        ],
        "bounds": [
            [5, 50],
            [0, 10],
            [2, 10],
            [0.001, 0.05],
            [0, 1],
            [0.5, 2.0],
        ],
    }
    return problem


def sample_parameter_space():
    param_values = sobol_sample.sample(problem(), 1028, calc_second_order=False)
    return param_values


def run_single_simulation(i, repeat, steps, recovery_rate, **kwargs):
    rng = np.random.default_rng(repeat)
    env_update_fn = piecewise_exponential_update(
        recovery=recovery_rate, pollution=1, gamma=0.01
    )
    model = VectorisedModel(rng=rng, env_update_fn=env_update_fn, **kwargs)
    model.run(steps)
    env_mean = np.mean(model.environment[-1])
    action_mean = np.mean(model.action[-1])
    pi_mean = metrics.pluralistic_ignorance(model).mean()
    clusters, _ = cluster_analysis.cluster_given_timestep(
        model, "environment", steps - 1
    )

    return i, repeat, env_mean, action_mean, pi_mean, clusters


def gather_output_statistics():
    repeats = 1
    steps = 1000
    param_values = sample_parameter_space()
    environment_output = np.empty((repeats, len(param_values)))
    action_output = np.empty_like(environment_output)
    pluralistic_ignorance = np.empty_like(environment_output)
    clusters_output = np.empty_like(environment_output)

    with ProcessPoolExecutor() as executor:
        futures = []

        for i, (
            width,
            rationality,
            memory_count,
            adaptation_speed,
            radius,
            recovery_rate,
        ) in enumerate(param_values):
            width = int(width)
            memory_count = int(memory_count)
            radius = np.round(radius)
            if radius == 1:
                radius_str = "all"
            elif radius == 0:
                radius_str = "single"
            else:
                raise ValueError("Unsupported radius value")

            kwargs = {
                "num_agents": width * width,
                "width": width,
                "height": width,
                "memory_count": memory_count,
                "rationality": rationality,
                "max_storage": steps,
                "moore": True,
                "simmer_time": 1,
                "neighb_prediction_option": "linear",
                "severity_benefit_option": "adaptive",
                "radius_option": radius_str,
                "prop_pessimistic": 1.0,
                "pessimism_level": 1.0,
                "gamma_s": adaptation_speed,
            }

            for r in range(repeats):
                future = executor.submit(
                    run_single_simulation, i, r, steps, recovery_rate, **kwargs
                )
                futures.append(future)

        for future in tqdm.tqdm(
            as_completed(futures), total=repeats * len(param_values)
        ):
            param_idx, repeat, n_mean, a_mean, pi_mean, clusters = future.result()
            environment_output[repeat, param_idx] = n_mean
            action_output[repeat, param_idx] = a_mean
            pluralistic_ignorance[repeat, param_idx] = pi_mean
            clusters_output[repeat, param_idx] = clusters

    environment_output = environment_output.mean(axis=0)
    action_output = action_output.mean(axis=0)
    pluralistic_ignorance = pluralistic_ignorance.mean(axis=0)
    clusters_output = clusters_output.mean(axis=0)

    np.savez(
        "data/output.npz",
        params=param_values,
        env=environment_output,
        act=action_output,
        pi=pluralistic_ignorance,
        cl=clusters_output,
    )
    return environment_output, action_output, pluralistic_ignorance, clusters_output


def plotting_output():
    load_data = np.load("data/output.npz")
    environment_output = load_data["env"]
    action_output = load_data["act"]
    pluralistic_ignorance = load_data["pi"]
    clusters_output = load_data["cl"]

    # Plot Environment Output
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(7, 2.5), constrained_layout=True, sharey=True
    )

    axes[0, 0].hist(
        environment_output, bins=50, color="steelblue", edgecolor="black", alpha=0.7
    )
    axes[0, 0].set_title("Environment Output", fontsize=12)
    # axes[0,0].set_xlabel("Mean Environment Value", fontsize=10)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylabel("Frequency", fontsize=10)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Plot Action Output
    axes[0, 1].hist(
        action_output, bins=50, color="darkorange", edgecolor="black", alpha=0.7
    )
    axes[0, 1].set_title("Action Output", fontsize=12)
    # axes[0,1].set_xlabel("Mean Action Value", fontsize=10)
    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Plot pluralistic ignorance
    axes[1, 0].hist(
        pluralistic_ignorance, bins=50, color="steelblue", edgecolor="black", alpha=0.7
    )
    axes[1, 0].set_title("Pluralistic ignorance", fontsize=12)
    # axes[1,0].set_xlabel("Pluralistic ignorance", fontsize=10)
    axes[1, 0].set_ylabel("Frequency", fontsize=10)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Plot numer of clusters
    axes[1, 1].hist(
        clusters_output, bins=50, color="gold", edgecolor="black", alpha=0.7
    )
    axes[1, 1].set_title("Number of clusters", fontsize=12)
    # axes[1,1].set_xlabel("Number of clusters", fontsize=10)
    axes[1, 1].set_xlim(0, 200)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    fig.savefig("plots/output.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_index(s, params, i, title="", method: str = "sobol", ax: Axes | None = None):
    ax = ax or plt.gca()
    ax.set_title(title)
    if method == "sobol":
        if i == "2":
            p = len(params)
            param_pairs = list(combinations(params, 2))
            indices = s["S" + i].reshape(p**2)
            indices = indices[~np.isnan(indices)]
            errors = s["S" + i + "_conf"].reshape(p**2)
            errors = errors[~np.isnan(errors)]
            labels = [f"{a}, {b}" for a, b in param_pairs]
        else:
            indices = s["S" + i]
            errors = s["S" + i + "_conf"]
            labels = params

        n_indices = len(indices)
        ax.set_ylim([-0.2, n_indices - 1 + 0.2])
        ax.set_yticks(range(n_indices), labels)
        ax.errorbar(
            indices, range(n_indices), xerr=errors, linestyle="None", marker="o"
        )
        ax.axvline(0, color="black")

    elif method == "pawn":
        medians = s["median"]
        labels = s["names"]
        ax.bar(labels, medians)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)

    else:
        raise ValueError(f"Unknown sensitivity analysis method: '{method}'")


def compute_sensitivity(method: str = "sobol"):
    load_data = np.load("data/output.npz")
    parameters = load_data["params"]
    environment_output = load_data["env"]
    action_output = load_data["act"]
    pluralistic_ignorance = load_data["pi"]
    clusters_output = load_data["cl"]

    problem_dict = problem()

    if method == "sobol":
        analyze = functools.partial(sobol.analyze, calc_second_order=False)
    elif method == "pawn":
        analyze = functools.partial(pawn.analyze, X=parameters)
    else:
        raise ValueError(f"Unknown sensitivity analysis method: '{method}'")

    results_environment = analyze(
        problem_dict,
        Y=environment_output,
    )
    results_action = analyze(
        problem_dict,
        Y=action_output,
    )
    results_pi = analyze(
        problem_dict,
        Y=pluralistic_ignorance,
    )

    results_cl = analyze(
        problem_dict,
        Y=clusters_output,
    )

    param_names = (
        "width",
        "rationality",
        "memory_count",
        "adaptation_speed",
        "radius",
        "recovery_rate",
    )

    if method == "sobol":
        fig, axes = plt.subplots(
            ncols=2, figsize=(5, 2.5), constrained_layout=True, sharey=True
        )
        plot_index(results_environment, param_names, "1", "First-order", ax=axes[0])
        plot_index(results_environment, param_names, "T", "Total-order", ax=axes[1])
        fig.suptitle("Environment")
        fig.savefig(
            f"plots/sensitivity_analysis_environment_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, axes = plt.subplots(
            ncols=2, figsize=(5, 2.5), constrained_layout=True, sharey=True
        )
        plot_index(results_action, param_names, "1", "First-order", ax=axes[0])
        plot_index(results_action, param_names, "T", "Total-order", ax=axes[1])
        fig.suptitle("Action")
        fig.savefig(
            f"plots/sensitivity_analysis_action_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, axes = plt.subplots(
            ncols=2, figsize=(5, 2.5), constrained_layout=True, sharey=True
        )
        plot_index(results_pi, param_names, "1", "First-order", ax=axes[0])
        plot_index(results_pi, param_names, "T", "Total-order", ax=axes[1])
        fig.suptitle("Pluralistic ignorance")
        fig.savefig(
            f"plots/sensitivity_analysis_pluralistic_ignorance_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, axes = plt.subplots(
            ncols=2, figsize=(5, 2.5), constrained_layout=True, sharey=True
        )
        plot_index(results_cl, param_names, "1", "First-order", ax=axes[0])
        plot_index(results_cl, param_names, "T", "Total-order", ax=axes[1])
        fig.suptitle("Number of clusters")
        fig.savefig(
            f"plots/sensitivity_analysis_cluster_number_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

    elif method == "pawn":
        fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
        plot_index(
            results_environment,
            param_names,
            "1",
            "PAWN (first-order)",
            method="pawn",
            ax=ax,
        )
        fig.suptitle("Environment")
        fig.savefig(
            f"plots/sensitivity_analysis_environment_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        plot_index(
            results_action, param_names, "1", "PAWN (first-order)", method="pawn", ax=ax
        )
        fig.suptitle("Action")
        fig.savefig(
            f"plots/sensitivity_analysis_action_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        plot_index(
            results_pi,
            param_names,
            "1",
            "PAWN (first-order)",
            method="pawn",
            ax=ax,
        )
        fig.suptitle("Pluralistic Ignorance")
        fig.savefig(
            f"plots/sensitivity_analysis_pluralistic_ignorance_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        plot_index(
            results_cl,
            param_names,
            "1",
            "PAWN (first-order)",
            method="pawn",
            ax=ax,
        )
        fig.suptitle("Number of Clusters")
        fig.savefig(
            f"plots/sensitivity_analysis_cluster_number_{method}.png",
            dpi=300,
            bbox_inches="tight",
        )

    return results_environment, results_action, pluralistic_ignorance, results_cl


if __name__ == "__main__":
    gather_output_statistics()
    plotting_output()
    # compute_sensitivity("sobol")
    # compute_sensitivity("pawn")

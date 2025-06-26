from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


# get parameter values
def problem():
    problem = {
        "num_vars": 3,
        "names": ["width", "rationality", "memory_count"],
        "bounds": [[5, 50], [0, 20], [1, 10]],
    }
    return problem


def sample_parameter_space():
    problem = {
        "num_vars": 3,
        "names": ["width", "rationality", "memory_count"],
        "bounds": [[5, 50], [0, 20], [1, 10]],
    }
    param_values = sobol_sample.sample(problem, 2)
    return param_values


def run_single_simulation(i, steps, **kwargs):
    rng = np.random.default_rng(i)
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    model = VectorisedModel(rng=rng, env_update_fn=env_update_fn, **kwargs)
    model.run(steps)
    env_mean = np.mean(model.environment[-1])
    action_mean = np.mean(model.action[-1])
    return i, env_mean, action_mean


def run_single_parameter_set(width, rationality, memory_count):
    num_runs = 15  # number of runs for the batch simulation
    steps = 10  # number of simulation steps

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
        "severity_benefit_option": None,
        "prop_pessimistic": 1.0,
        "pessimism_level": 1.0,
    }

    env_means = np.zeros((num_runs, 1))
    # env_vars = np.zeros((steps + 1,))
    action_means = np.zeros((num_runs, 1))
    # action_vars = np.zeros((steps + 1,))

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_simulation, i, steps, **kwargs)
            for i in range(num_runs)
        ]
        for future in as_completed(futures):
            i, n_mean, a_mean = future.result()
            env_means[i] = n_mean
            action_means[i] = a_mean

    mean_env = np.mean(env_means)
    mean_action = np.mean(action_means)

    return mean_env, mean_action


def gather_output_statistics():
    param_values = sample_parameter_space()
    environment_output = []
    action_output = []

    for width, rationality, memory_count in tqdm.tqdm(param_values):
        width = int(width)
        memory_count = int(memory_count)
        mean_env, mean_action = run_single_parameter_set(
            width, rationality, memory_count
        )
        environment_output.append(mean_env)
        action_output.append(mean_action)
    np.savez("data/output.npz", env=environment_output, act=action_output)
    return environment_output, action_output


def plotting_output():
    load_data = np.load("data/output.npz")
    environment_output = load_data["env"]
    action_output = load_data["act"]

    # Plot Environment Output
    plt.figure(figsize=(8, 5))
    plt.hist(
        environment_output, bins=50, color="steelblue", edgecolor="black", alpha=0.7
    )
    plt.title("Environment Output", fontsize=14)
    plt.xlabel("Mean Environment Value", fontsize=12)
    plt.xlim(0, 1)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/environment_output.png")
    plt.show()

    # Plot Action Output
    plt.figure(figsize=(8, 5))
    plt.hist(action_output, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
    plt.title("Action Output", fontsize=14)
    plt.xlabel("Mean Action Value", fontsize=12)
    plt.xlim(-1, 1)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/action_output.png")
    plt.show()


def plot_index(s, params, i, title=""):
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
        plt.figure()

    n_indices = len(indices)
    plt.title(title)
    plt.ylim([-0.2, n_indices - 1 + 0.2])
    plt.yticks(range(n_indices), labels)
    plt.errorbar(indices, range(n_indices), xerr=errors, linestyle="None", marker="o")
    plt.axvline(0, color="black")
    plt.tight_layout()


def sobol_sensitivity():
    load_data = np.load("data/output_environment.npz")
    environment_output = load_data["env"]
    action_output = load_data["act"]

    problem_dict = problem()

    sobol_environment = sobol.analyze(problem_dict, environment_output)
    sobol_action = sobol.analyze(problem_dict, action_output)

    param_names = ("width", "rationality", "memory_count")

    for Si, label in zip(
        [sobol_environment, sobol_action],
        ["Environment output", "Action output"],
        strict=False,
    ):
        plot_index(Si, param_names, "1", f"First-order sensitivity ({label})")
        plt.savefig(f"plots/S1_{label}")
        plt.show()

        plot_index(Si, param_names, "2", f"Second-order sensitivity ({label})")
        plt.savefig(f"plots/S2_{label}")
        plt.show()

        plot_index(Si, param_names, "T", f"Total-order sensitivity ({label})")
        plt.savefig(f"plots/ST_{label}")
        plt.show()

    return sobol_environment, sobol_action


if __name__ == "__main__":
    gather_output_statistics()
    plotting_output()
    sobol_sensitivity()

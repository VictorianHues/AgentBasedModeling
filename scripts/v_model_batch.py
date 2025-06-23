"""Batch script to run the vectorised model and generate animations of the results."""

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

from abm_project.batch_run_tools import (
    clustering_score_over_time,
)
from abm_project.plotting import (
    get_data_directory,
    plot_list_over_time,
    plot_mean_and_variability_array,
    plot_sobol_indices,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def define_problem():
    return {
        "num_vars": 3,
        "names": ["rationality", "prop_pessimistic", "pessimism_level"],
        "bounds": [
            [0.1, 10.0],
            [0.0, 1.0],
            [1.0, 3.0],
        ],
    }


def run_vectorised_model(rationality, prop_pessimistic, pessimism_level):
    width = 50
    height = 50
    num_agents = int(width * height)
    num_steps = 100
    memory_count = 1
    simmer_time = 1
    moore_option = True
    rng = None
    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        env_update_fn=piecewise_exponential_update(alpha=1, beta=1, rate=0.01),
        rng=rng,
        rationality=rationality,
        max_storage=num_steps,
        moore=moore_option,
        simmer_time=simmer_time,
        neighb_prediction_option="logistic",
        severity_benefit_option="adaptive",
        prop_pessimistic=prop_pessimistic,
        pessimism_level=pessimism_level,
    )
    model.run(steps=num_steps)
    print(model.environment[-1])
    return model


def run_sobol_analysis():
    sample_num = 50
    time_steps = [10, 100]

    problem = define_problem()
    param_values = saltelli.sample(problem, sample_num)

    print("Running vectorised model with Sobol sampling...")

    Y = np.zeros((len(param_values), len(time_steps)))
    Z = np.zeros((len(param_values), len(time_steps)))
    for i, (rationality, prop_pessimistic, pessimism_level) in enumerate(param_values):
        model = run_vectorised_model(rationality, prop_pessimistic, pessimism_level)
        for j, t in enumerate(time_steps):
            Y[i, j] = np.mean(model.environment[t - 1])
            Z[i, j] = np.mean(model.action[t - 1])
        print(
            f"Run {i + 1}/{len(param_values)}: rationality={rationality}, "
            f"prop_pessimistic={prop_pessimistic}, "
            f"pessimism_level={pessimism_level}, "
            f"Y={Y[i]}, and Z={Z[i]}"
        )

    print("Y min:", np.min(Y), "Y max:", np.max(Y), "Y std:", np.std(Y))
    print("Z min:", np.min(Z), "Z max:", np.max(Z), "Z std:", np.std(Z))

    np.savez("sobol_results.npz", param_values=param_values, Y=Y, time_steps=time_steps)
    np.savez(
        "sobol_results_Z.npz", param_values=param_values, Z=Z, time_steps=time_steps
    )

    Si_Y = {}
    Si_Z = {}

    for j, t in enumerate(time_steps):
        Si_Y_t = sobol.analyze(problem, Y[:, j], print_to_console=True)
        Si_Z_t = sobol.analyze(problem, Z[:, j], print_to_console=True)
        Si_Y[t] = {"S1": Si_Y_t["S1"], "ST": Si_Y_t["ST"], "S2": Si_Y_t["S2"]}
        Si_Z[t] = {"S1": Si_Z_t["S1"], "ST": Si_Z_t["ST"], "S2": Si_Z_t["S2"]}

    return Si_Y, Si_Z, time_steps


def run_single_parameter_set():
    num_runs = 1000  # number of runs for the batch simulation
    steps = 1000  # number of simulation steps

    kwargs = {
        "num_agents": 2500,
        "width": 50,
        "height": 50,
        "memory_count": 1,
        "env_update_fn": piecewise_exponential_update(alpha=1, beta=1, rate=0.01),
        "rng": None,
        "rationality": 10.0,
        "max_storage": steps,
        "moore": True,
        "simmer_time": 1,
        "neighb_prediction_option": "logistic",
        "severity_benefit_option": None,
        "prop_pessimistic": 1.0,
        "pessimism_level": 1.0,
    }

    env_means = np.zeros((num_runs, steps + 1))
    env_vars = np.zeros((steps + 1,))
    action_means = np.zeros((num_runs, steps + 1))
    action_vars = np.zeros((steps + 1,))
    action_cluster_scores = np.zeros(
        (
            num_runs,
            steps + 1,
        )
    )

    for run in range(num_runs):
        model = VectorisedModel(**kwargs)
        model.run(steps)

        env_means[run, :] = np.mean(model.environment, axis=1)
        action_means[run, :] = np.mean(model.action, axis=1)
        action_cluster_scores[run, :] = clustering_score_over_time(
            model,
            attribute="action",
            width=kwargs["width"],
            height=kwargs["height"],
            radius=1,
        )

        del model

    data_file_path = get_data_directory("batch_run_results.npz")
    np.savez(
        data_file_path,
        env_means=env_means,
        action_means=action_means,
        cluster_scores=action_cluster_scores,
    )

    plot_mean_and_variability_array(
        env_means, title="Environment Over Time", file_name="env_over_time.png"
    )
    plot_mean_and_variability_array(
        action_means, title="Actions Over Time", file_name="action_over_time.png"
    )
    plot_mean_and_variability_array(
        action_cluster_scores,
        title="Spatial Clustering Over Time",
        file_name="action_cluster_scores_over_time.png",
    )

    env_vars = np.var(env_means, axis=0)
    action_vars = np.var(action_means, axis=0)
    action_cluster_score_vars = np.var(action_cluster_scores, axis=0)

    plot_list_over_time(
        env_vars,
        title="Variance of Environment Status Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="env_variance_over_time.png",
    )
    plot_list_over_time(
        action_vars,
        title="Variance of Actions Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="action_variance_over_time.png",
    )
    plot_list_over_time(
        action_cluster_score_vars,
        title="Variance of Action Cluster Scores Over Time",
        xlabel="Time Step",
        ylabel="Variance",
        file_name="action_cluster_score_variance_over_time.png",
    )


def sobol_analysis():
    problem = define_problem()
    results_env, results_act, time_steps = run_sobol_analysis()

    print("Sobol analysis results for environment:")
    for t, res in results_env.items():
        print(f"Time step {t}: S1={res['S1']}, ST={res['ST']}, S2={res['S2']}")

    print("Sobol analysis results for actions:")
    for t, res in results_act.items():
        print(f"Time step {t}: S1={res['S1']}, ST={res['ST']}, S2={res['S2']}")

    plot_sobol_indices(
        results_env,
        time_steps,
        problem["names"],
        "Environment",
        file_name="sobol_env_indices.png",
    )


if __name__ == "__main__":
    run_single_parameter_set()
    sobol_analysis()

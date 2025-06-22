"""Base model script for running an agent-based model simulation."""

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

from abm_project.batch_run_tools import (
    attribute_variance_over_time,
    run_batch,
    spatial_clustering_over_time,
)
from abm_project.oop_model import BaseModel
from abm_project.plotting import plot_list_over_time, plot_mean_with_variability


def run_model_output(radius, memory_count, peer_pressure_learning_rate, rationality):
    """Run the agent-based model simulation with given parameters."""
    rng = None

    def env_status_fn():
        if rng:
            return rng.uniform(0.01, 0.99)
        else:
            return np.random.uniform(0.01, 0.99)

    def peer_pressure_coeff_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    def env_perception_coeff_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    model = BaseModel(
        width=50,
        height=50,
        radius=radius,
        memory_count=memory_count,
        env_update_option="linear",
        adaptive_attr_option=None,
        neighb_prediction_option="logistic",
        peer_pressure_learning_rate=peer_pressure_learning_rate,
        rationality=rationality,
        rng=None,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )
    model.run(steps=1000)
    final_action_grid = model.agent_action_history[-1]
    print(
        f"Simulation completed with radius={radius}, "
        f"memory_count={memory_count}, "
        f"peer_pressure_learning_rate={peer_pressure_learning_rate}, "
        f"rationality={rationality}. "
        f"Final action grid mean: {np.mean(final_action_grid)}"
    )
    return np.mean(final_action_grid)


def run_full_sobol():
    """Main function to run the agent-based model simulation."""
    problem = {
        "num_vars": 4,
        "names": [
            "radius",
            "memory_count",
            "peer_pressure_learning_rate",
            "rationality",
        ],
        "bounds": [[1, 24], [1, 10], [0.05, 0.3], [0.0, 1.0]],
    }

    param_values = saltelli.sample(problem, 128)  # number of base samples

    Y = np.zeros(len(param_values))
    for i, (rad, mem, lr_peer, rationality) in enumerate(param_values):
        Y[i] = run_model_output(int(rad), int(mem), lr_peer, int(rationality))

    np.save("sobol_output.npy", Y)

    Si = sobol.analyze(problem, Y)

    print("First-order:", Si["S1"])
    print("Total-order:", Si["ST"])
    print("Second-order:", Si["S2"])

    plt.bar(problem["names"], Si["S1"], label="First-order")
    plt.bar(problem["names"], Si["ST"], bottom=Si["S1"], alpha=0.5, label="Total-order")
    plt.ylabel("Sensitivity Index")
    plt.legend()
    plt.title("Sobol Sensitivity Indices")
    plt.show()


def run_single_parameter_set():
    """Run a single parameter set for the agent-based model simulation."""
    kwargs = {
        "width": 50,
        "height": 50,
        "radius": 1,
        "memory_count": 1,
        "env_update_option": "linear",
        "adaptive_attr_option": "bayesian",
        "neighb_prediction_option": "logistic",
        "peer_pressure_learning_rate": 0.0,
        "rationality": 1.0,
        "env_status_fn": lambda: np.random.uniform(0.0, 1.0),
        "peer_pressure_coeff_fn": lambda: np.random.uniform(1.0, 1.0),
        "env_perception_coeff_fn": lambda: np.random.uniform(1.0, 1.0),
    }

    models = run_batch(num_runs=100, model_class=BaseModel, steps=1000, **kwargs)

    # peer_pressure_variance = attribute_variance_over_time(
    #     models,
    #     'agent_peer_pressure_coeff_history'
    # )
    env_status_variance = attribute_variance_over_time(
        models, "agent_env_status_history"
    )
    # env_perception_variance = attribute_variance_over_time(
    #     models,
    #     'agent_env_perception_coeff_history'
    # )

    env_status_clustering_score_over_times = spatial_clustering_over_time(
        models, radius=1
    )

    plot_mean_with_variability(
        models,
        "agent_env_status_history",
        "Environment Status over Time",
        "Environmental Status",
    )
    plot_mean_with_variability(
        models,
        "agent_peer_pressure_coeff_history",
        "Peer Pressure Coefficient over Time",
        "Peer Pressure Coefficient",
    )

    plot_list_over_time(
        env_status_variance,
        title="Variance of Environment Status Over Time",
        xlabel="Time Step",
        ylabel="Variance",
    )

    plot_list_over_time(
        env_status_clustering_score_over_times,
        title="Spatial Clustering Score Over Time",
        xlabel="Time Step",
        ylabel="Clustering Score",
    )


if __name__ == "__main__":
    run_full_sobol()
    run_single_parameter_set()

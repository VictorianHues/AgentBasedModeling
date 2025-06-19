"""Base model script for running an agent-based model simulation."""

import numpy as np
from matplotlib.colors import ListedColormap

from abm_project.oop_model import BaseModel
from abm_project.plotting import (
    animate_grid_states,
    plot_current_grid_state,
    plot_grid_average_over_time,
)


def main():
    """Main function to run the agent-based model simulation."""
    num_agents = 2500
    width = 50
    height = 50
    radius = 1
    num_steps = 100
    memory_count = 10

    # "linear", "sigmoid", "exponential", "bell", "sigmoid_asymmetric", "bimodal"
    env_update_option = "linear"

    # "bayesian_niegh_utility", "bayesian_niegh_action"
    adaptive_attr_option = "bayesian_niegh_utility"
    rng = None

    def env_status_fn():
        if rng:
            return rng.uniform(0.0, 0.5)
        else:
            return np.random.uniform(0.0, 0.5)

    def peer_pressure_coeff_fn():
        if rng:
            return rng.uniform(0.5, 0.5)
        else:
            return np.random.uniform(0.5, 0.5)

    def env_perception_coeff_fn():
        if rng:
            return rng.uniform(0.5, 0.5)
        else:
            return np.random.uniform(0.5, 0.5)

    model = BaseModel(
        num_agents=num_agents,
        width=width,
        height=height,
        radius=radius,
        memory_count=memory_count,
        env_update_option=env_update_option,
        adaptive_attr_option=adaptive_attr_option,
        rng=rng,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )

    model.run(num_steps)

    plot_current_grid_state(
        model.get_agent_env_status(),
        colormap="RdYlGn",
        title="Agent Environment Status",
        colorbar_label="Environment Status",
        clim=(0, 1),
    )
    plot_current_grid_state(
        model.get_agent_actions(),
        colormap=ListedColormap(["red", "green"]),
        title="Agent Actions",
        colorbar_label="Agent Action (-1 or 1)",
        clim=(-1, 1),
    )

    animate_grid_states(
        np.array(model.agent_action_history),
        colormap=ListedColormap(["red", "green"]),
        title="Agent Actions Over Time",
        colorbar_label="Agent Action (-1 or 1)",
        file_name="agent_actions.mp4",
        clim=(-1, 1),
    )
    animate_grid_states(
        np.array(model.agent_env_status_history),
        colormap="RdYlGn",
        title="Agent Environment Status Over Time",
        colorbar_label="Environment Status",
        file_name="agent_env_status.mp4",
        clim=(0, 1),
    )
    animate_grid_states(
        np.array(model.agent_peer_pressure_coeff_history),
        colormap="viridis",
        title="Agent Peer Pressure Coeff Over Time",
        colorbar_label="Peer Pressure Coeff",
        file_name="agent_peer_pressure_coeff.mp4",
        clim=(0, 1),
    )

    plot_grid_average_over_time(
        np.array(model.agent_peer_pressure_coeff_history),
        title="Overall Agent Peer Pressure Coeff Over Time",
        xlabel="Time Step",
        ylabel="Peer Pressure Coeff",
        file_name="overall_agent_peer_pressure_coeff_over_time.png",
    )
    plot_grid_average_over_time(
        np.array(model.agent_env_status_history),
        title="Overall Agent Environment Status Over Time",
        xlabel="Time Step",
        ylabel="Environment Status",
        file_name="overall_agent_env_status_over_time.png",
    )
    plot_grid_average_over_time(
        np.array(model.agent_action_history),
        title="Overall Agent Actions Over Time",
        xlabel="Time Step",
        ylabel="Agent Action (-1 or 1)",
        file_name="overall_agent_actions_over_time.png",
    )
    plot_grid_average_over_time(
        np.array(model.agent_env_utility_history),
        title="Overall Agent Environment Utility Over Time",
        xlabel="Time Step",
        ylabel="Environment Utility",
        file_name="overall_agent_env_utility_over_time.png",
    )


if __name__ == "__main__":
    main()

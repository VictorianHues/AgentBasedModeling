"""Base model script for running an agent-based model simulation."""

import numpy as np

from abm_project.oop_model import BaseModel
from abm_project.plotting import (
    animate_agent_actions,
    animate_agent_env_status,
    plot_current_agent_actions,
    plot_current_agent_env_status,
    plot_overall_agent_actions_over_time,
    plot_overall_agent_env_status_over_time,
)


def main():
    """Main function to run the agent-based model simulation."""
    num_agents = 25
    width = 5
    height = 5
    radius = 1
    # num_steps = 200
    memory_count = 20
    forecast_threshold = 0.6
    env_update_option = "linear"
    rng = None

    def env_status_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    def peer_pressure_coeff_fn():
        if rng:
            return rng.uniform(0, 1)
        else:
            return np.random.uniform(0.5, 1.0)

    def env_perception_coeff_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    model = BaseModel(
        num_agents=num_agents,
        width=width,
        height=height,
        radius=radius,
        memory_count=memory_count,
        rng=rng,
        forecast_threshold=forecast_threshold,
        env_update_option=env_update_option,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )

    plot_current_agent_env_status(model)
    plot_current_agent_actions(model)

    animate_agent_env_status(model, file_name="agent_env_status.mp4")
    animate_agent_actions(model, file_name="agent_actions.mp4")
    plot_overall_agent_actions_over_time(
        model, file_name="overall_agent_actions_over_time.png"
    )
    plot_overall_agent_env_status_over_time(
        model, file_name="overall_agent_env_status_over_time.png"
    )

    # plot_agent_action_distribution(
    #     model, file_name="agent_action_distribution.png"
    # )
    # plot_cooperator_phase_plot(
    #     model, file_name="cooperator_phase_plot.png"
    # )
    # plot_environment_variance_over_time(
    #     model, file_name="env_variance_over_time.png"
    # )


if __name__ == "__main__":
    main()

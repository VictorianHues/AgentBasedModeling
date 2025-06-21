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
    num_agents = 2500
    width = 50
    height = 50
    radius = 1
    num_steps = 500
    memory_count = num_steps
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
        env_update_option=env_update_option,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )

    model.run(num_steps)
    print(f"Simulation completed with {num_steps} steps.\n")

    plot_current_agent_env_status(model)
    plot_current_agent_actions(model)

    animate_agent_env_status(
        model, file_name="full_memory_steps_500/agent_env_status.mp4"
    )
    animate_agent_actions(model, file_name="full_memory_steps_500/agent_actions.mp4")
    plot_overall_agent_actions_over_time(
        model, file_name="full_memory_steps_500/overall_agent_actions_over_time.png"
    )
    plot_overall_agent_env_status_over_time(
        model, file_name="full_memory_steps_500/overall_agent_env_status_over_time.png"
    )


if __name__ == "__main__":
    main()

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
    num_agents = 400
    width = 20
    height = 20
    radius = 1
    num_steps = 1000
    memory_count = 10
    rng = None

    def env_status_fn():
        if rng:
            return rng.uniform(-1, 1)
        else:
            return np.random.uniform(0, 1)

    def peer_pressure_coeff_fn():
        if rng:
            return rng.uniform(0, 1)
        else:
            return np.random.uniform(0.0, 0.1)

    def env_perception_coeff_fn():
        if rng:
            return rng.uniform(0.1, 1.0)
        else:
            return np.random.uniform(0.1, 1.0)

    model = BaseModel(
        num_agents=num_agents,
        width=width,
        height=height,
        radius=radius,
        memory_count=memory_count,
        rng=rng,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )

    model.run(num_steps)

    plot_current_agent_env_status(model)
    plot_current_agent_actions(model)

    animate_agent_env_status(model)
    animate_agent_actions(model)
    plot_overall_agent_actions_over_time(model)
    plot_overall_agent_env_status_over_time(model)


if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""Base model script for running an agent-based model simulation."""

from abm_project.oop_model import BaseModel
from abm_project.plotting import (
    animate_agent_actions,
    animate_agent_env_status,
    plot_current_agent_actions,
    plot_current_agent_env_status,
)


def main():
    """Main function to run the agent-based model simulation."""
    num_agents = 400
    width = 20
    height = 20
    radius = 1
    num_steps = 25

    model = BaseModel(num_agents=num_agents, width=width, height=height, radius=radius,  severity_type="Arrow_Pratt")

    model.run(num_steps)

    plot_current_agent_env_status(model)
    plot_current_agent_actions(model)

    animate_agent_env_status(model)
    animate_agent_actions(model)


if __name__ == "__main__":
    main()

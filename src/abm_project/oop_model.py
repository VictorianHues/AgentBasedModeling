import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Agent-based model base class for simulations.

This module defines a base class for agent-based models, providing
a framework for initializing agents, calculating neighbor actions,
and managing the simulation environment.

"""

import numpy as np
from abm_project.agent import Agent


class BaseModel:
    """Base model for agent-based simulations.

    This model initializes a grid of agents and provides methods for
    neighbor calculations, agent actions, and environment status updates.
    Agents can decide their actions based on the average actions of their neighbors.
    The model supports a Moore neighborhood for neighbor calculations.

    Attributes:
        time (int): Current time step in the simulation.
        num_agents (int): Total number of agents in the grid.
        radius (int): Radius for neighbor calculations.
        width (int): Width of the grid.
        height (int): Height of the grid.
        rng (np.random.Generator):
            Random number generator for stochastic processes.
        agents (np.ndarray):
            2D array of Agent objects representing the grid.
        agent_action_history (list):
            History of agent actions at each time step.
        agent_env_status_history (list):
            History of agent environment status at each time step.
    """

    DEFAULT_NUM_AGENTS = 100
    DEFAULT_WIDTH = 10
    DEFAULT_HEIGHT = 10
    DEFAULT_RADIUS = 1

    def __init__(
        self,
        num_agents: int = DEFAULT_NUM_AGENTS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        radius: int = DEFAULT_RADIUS,
        rng: np.random.Generator = None,
        env_status_fn=None,
        peer_pressure_coeff_fn=None,
        env_perception_coeff_fn=None,
        risk_aversion_coeff_fn = None,
        severity_type= "Arrow_Pratt"
    ):
        """Initialize the base model with a grid of agents.

        Args:
            num_agents (int): Total number of agents in the grid.
            width (int): Width of the grid.
            height (int): Height of the grid.
            radius (int): Radius for neighbor calculations.
            rng (np.random.Generator, optional):
                Random number generator. Defaults to None.
            env_status_fn (callable, optional):
                Function to initialize env_status.
            peer_pressure_coeff_fn (callable, optional):
                Function to initialize peer_pressure_coeff.
            env_perception_coeff_fn (callable, optional):
                Function to initialize env_perception_coeff.
            risk_aversion_coeff_fn (callable, optional):
                Function to initialize risk_aversion_coeff.
        """
        self.time = 0
        self.num_agents = num_agents
        self.radius = radius
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()

        self.agents = np.empty((width, height), dtype=object)
        i = 0
        for x in range(self.width):
            for y in range(self.height):
                if env_status_fn is not None:
                    env_status = env_status_fn(x, y, i, self.rng)
                else:
                    env_status = self.rng.uniform(0.0, 1.0)
                if peer_pressure_coeff_fn is not None:
                    peer_pressure_coeff = peer_pressure_coeff_fn(x, y, i, self.rng)
                else:
                    peer_pressure_coeff = self.rng.uniform(0.0, 1.0)
                if env_perception_coeff_fn is not None:
                    env_perception_coeff = env_perception_coeff_fn(x, y, i, self.rng)
                else:
                    env_perception_coeff = self.rng.uniform(0.0, 1.0)
                
                if risk_aversion_coeff_fn is not None:
                    risk_aversion_coeff = risk_aversion_coeff_fn(x, y, i, self.rng)
                else:
                    risk_aversion_coeff = self.rng.uniform(-0.5, 0.5)

                self.agents[x, y] = Agent(
                    i, env_status, peer_pressure_coeff, env_perception_coeff, risk_aversion_coeff, severity_type, self.rng 
                )

                i += 1

        self.agent_action_history = [self.get_agent_actions()]
        self.agent_env_status_history = [self.get_agent_env_status()]

    def get_neighbors(self, x: int, y: int) -> list[Agent]:
        """Get the neighbors of an agent at position (x, y).

        Neighbors are defined as agents within a Moore
        neighborhood depending on the radius.

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            List[Agent]: List of neighboring agents.
        """
        neighbors = []
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                neighbors.append(self.agents[nx, ny])

        return neighbors

    def ave_neighb_action_single_memory(self, x: int, y: int) -> float:
        """Calculate the average action of peers based on their most recent action.

        This method computes the average action of neighboring agents,
        considering only their most recent action.

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            float: Average action of neighbors.
        """
        total_action = 0
        count = 0
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor.past_actions:
                total_action += neighbor.past_actions[-1]
                count += 1
        return total_action / count if count > 0 else 0

    def ave_neighb_action_full_memory(self, x: int, y: int) -> float:
        """Calculate the average action of peers based on their full action history.

        This method averages all past actions of neighbors,
        not just the most recent one.

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            float: Average action of neighbors.
        """
        total_action = 0
        count = 0
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor.past_actions:
                total_action += np.mean(neighbor.past_actions)
                count += 1
        return total_action / count if count > 0 else 0

    def step(self) -> None:
        """Perform a single step in the model.

        This method updates the environment status and allows
        agents to decide their actions.
        """
        self.time += 1

        for x in range(self.width):
            for y in range(self.height):
                agent = self.agents[x, y]
                ave_peer_action = self.ave_neighb_action_single_memory(x, y)
                agent.decide_action(ave_peer_action)

    def get_agent_actions(self) -> np.ndarray:
        """Get the actions of all agents in a 2D array.

        Returns:
            np.ndarray: A 2D array where each cell contains the
            action of the agent at that position.
        """
        action_grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                action_grid[x, y] = self.agents[x, y].action
        return action_grid

    def get_agent_env_status(self) -> np.ndarray:
        """Get the environment status of all agents in a 2D array.

        Returns:
            np.ndarray: A 2D array where each cell contains the
            environment status of the agent at that position.
        """
        env_status_grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                env_status_grid[x, y] = self.agents[x, y].env_status
        return env_status_grid

    def run(self, steps: int = 20) -> None:
        """Run the model for a specified number of steps.

        Args:
            steps (int): Number of steps to run the model.
        """
        for _ in range(steps):
            self.step()
            self.agent_action_history.append(self.get_agent_actions())
            self.agent_env_status_history.append(self.get_agent_env_status())

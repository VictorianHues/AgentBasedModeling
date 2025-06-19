"""Agent-based model base class for simulations.

This module defines a base class for agent-based models, providing
a framework for initializing agents, calculating neighbor actions,
and managing the simulation environment.

"""

import os

import numpy as np

from abm_project.agent import Agent


class BaseModel:
    """Base model for agent-based simulations.

    This class initializes a grid of agents and provides methods
    for running the simulation, updating the environment, and
    calculating neighbor actions.

    Attributes:
        num_agents (int): Total number of agents in the grid.
        width (int): Width of the grid.
        height (int): Height of the grid.
        radius (int): Radius for neighbor calculations.
        memory_count (int): Number of past actions to remember for each agent.
        env_update_option (str): Method to update the environment status.
        adaptive_attr_option (str): Option for adaptive attributes.
        rng (np.random.Generator): Random number generator.
        agents (np.ndarray): 2D array of Agent objects representing the grid.
        agent_action_history (list): History of agent actions.
        agent_env_status_history (list): History of environment status.
        agent_peer_pressure_coeff_history (list): History of peer pressure coefficients.
        agent_env_utility_history (list): History of environment utilities.
        time (int): Current time step in the simulation.
    """

    DEFAULT_NUM_AGENTS = 100
    DEFAULT_WIDTH = 10
    DEFAULT_HEIGHT = 10
    DEFAULT_RADIUS = 1
    DEFAULT_MEMORY_COUNT = 1
    DEFAULT_ENV_UPDATE_OPTION = "linear"
    DEFAULT_ADAPTIVE_ATTR_OPTION = None
    DEFAULT_LEARNING_RATE = 0.1

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        radius: int = DEFAULT_RADIUS,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
        adaptive_attr_option: str = DEFAULT_ADAPTIVE_ATTR_OPTION,
        env_perception_learning_rate: float = DEFAULT_LEARNING_RATE,
        peer_pressure_learning_rate: float = DEFAULT_LEARNING_RATE,
        rng: np.random.Generator = None,
        env_status_fn=None,
        peer_pressure_coeff_fn=None,
        env_perception_coeff_fn=None,
        results_save_name: str = None,
    ):
        """Initialize the base model with a grid of agents.

        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.
            radius (int): Radius for neighbor calculations.
            memory_count (int): Number of past actions to remember for each agent.
            env_update_option (str, optional):
                Method to update the environment status.
            adaptive_attr_option (str, optional):
                Option for adaptive attributes. Defaults to None.
            env_perception_learning_rate (float, optional):
                Learning rate for environment perception updates.
            peer_pressure_learning_rate (float, optional):
                Learning rate for peer pressure updates.
            results_save_name (str, optional):
                Name for saving results. If None, results are not saved.
            rng (np.random.Generator, optional):
                Random number generator. Defaults to None.
            env_status_fn (callable, optional):
                Function to initialize env_status.
            peer_pressure_coeff_fn (callable, optional):
                Function to initialize peer_pressure_coeff.
            env_perception_coeff_fn (callable, optional):
                Function to initialize env_perception_coeff.
        """
        self.time = 0
        self.radius = radius
        self.width = width
        self.height = height
        self.memory_count = memory_count
        self.env_update_option = env_update_option
        self.adaptive_attr_option = adaptive_attr_option
        self.env_perception_learning_rate = env_perception_learning_rate
        self.peer_pressure_learning_rate = peer_pressure_learning_rate

        self.rng = rng or np.random.default_rng()
        self.results_save_name = results_save_name

        self.agents = np.empty((width, height), dtype=object)
        i = 0
        for x in range(self.width):
            for y in range(self.height):
                self.agents[x, y] = Agent(
                    i,
                    self.memory_count,
                    self.rng,
                    self.env_update_option,
                    self.adaptive_attr_option,
                    self.env_perception_learning_rate,
                    self.peer_pressure_learning_rate,
                    env_status_fn,
                    peer_pressure_coeff_fn,
                    env_perception_coeff_fn,
                )
                i += 1

        self.agent_action_history = [self.get_agent_grid_attribute("past_actions")]
        self.agent_env_status_history = [self.get_agent_grid_attribute("env_status")]
        self.agent_peer_pressure_coeff_history = [
            self.get_agent_grid_attribute("peer_pressure_coeff")
        ]
        self.agent_env_perception_coeff_history = [
            self.get_agent_grid_attribute("env_perception_coeff")
        ]
        self.agent_env_utility_history = [
            self.get_agent_grid_attribute("env_utility_history")
        ]

    def step(self) -> None:
        """Perform a single step in the model.

        This method updates the environment status and allows
        agents to decide their actions.
        """
        self.time += 1

        for x in range(self.width):
            for y in range(self.height):
                agent = self.agents[x, y]
                ave_peer_action = self.ave_neighb_action(x, y, memory=self.memory_count)
                all_peer_actions = self.get_neighbor_attribute_values(
                    x, y, "past_actions"
                )
                agent.decide_action(ave_peer_action, all_peer_actions)

    def run(self, steps: int = 20) -> None:
        """Run the model for a specified number of steps.

        Args:
            steps (int): Number of steps to run the model.
        """
        for _ in range(steps):
            self.step()
            self.agent_action_history.append(
                self.get_agent_grid_attribute("past_actions")
            )
            self.agent_env_status_history.append(
                self.get_agent_grid_attribute("env_status")
            )
            self.agent_peer_pressure_coeff_history.append(
                self.get_agent_grid_attribute("peer_pressure_coeff")
            )
            self.agent_env_perception_coeff_history.append(
                self.get_agent_grid_attribute("env_perception_coeff")
            )
            self.agent_env_utility_history.append(
                self.get_agent_grid_attribute("env_utility_history")
            )

        self.save_results()

    ###################################################################################################################
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

    def ave_neighb_action(self, x: int, y: int, memory: int = 1) -> float:
        """Calculate the average action of peers based on their recent actions.

        This method computes the average action of neighboring agents,
        considering the last `memory` actions (from the end). If memory=1,
        only the most recent action is used. If memory > 1, the mean of the
        last `memory` actions is used for each neighbor.

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.
            memory (int): Number of most recent actions to consider.

        Returns:
            float: Average action of neighbors.
        """
        total_action = 0
        count = 0
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor.past_actions:
                if memory == 1:
                    value = neighbor.past_actions[-1]
                else:
                    value = np.mean(neighbor.past_actions[-memory:])
                total_action += value
                count += 1
        return total_action / count if count > 0 else 0

    def get_neighbor_attribute_values(
        self, x: int, y: int, attribute: str
    ) -> np.ndarray:
        """Get the values of a specific attribute from neighboring agents.

        This method retrieves the most recent values of a specified
        attribute from all neighbors of the agent at position (x, y).

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.
            attribute (str): The attribute to retrieve from neighbors.

        Returns:
            np.ndarray: Array of attribute values from neighboring agents.
        """
        neighbor_values = []
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            value = getattr(neighbor, attribute, [])
            if isinstance(value, list | np.ndarray):
                neighbor_values.append(value[-1] if value else 0)
            else:
                neighbor_values.append(value if value is not None else 0)
        return np.array(neighbor_values)

    def get_agent_grid_attribute(self, attribute: str) -> np.ndarray:
        """Get a 2D array of a specific agent attribute or method result.

        Args:
            attribute (str): The attribute or method name to retrieve from agents.

        Returns:
            np.ndarray: A 2D array with values from the specified attribute or method.
        """
        grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                agent = self.agents[x, y]
                value = getattr(agent, attribute)
                grid[x, y] = value[-1] if len(value) > 0 else 0
        return grid

    def get_agent_attribute_at_time(self, attribute: str, time: int) -> np.ndarray:
        """Get the values of a specific agent attribute at a given time step.

        Args:
            attribute (str):
                The attribute history to retrieve (e.g., 'agent_action_history').
            time (int):
                The time step to retrieve values for.

        Returns:
            np.ndarray: A 2D array of the attribute values at the specified time.
        """
        history = getattr(self, attribute, None)
        if history is None:
            raise AttributeError(f"No such attribute history: {attribute}")
        if time < len(history):
            return history[time]
        else:
            raise IndexError("Time step exceeds the history length.")

    def save_results(self) -> None:
        """Save the agent history to a file.

        This method saves the actions, environment status,
        peer pressure coefficients, and environment utilities
        of agents to a .npz file for later analysis.
        """
        if self.results_save_name is None:
            return

        file_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if self.results_save_name.endswith(".npz"):
            file_path = os.path.join(file_dir, self.results_save_name)
        else:
            # Ensure the file name ends with .npz
            self.results_save_name += ".npz"
            file_path = os.path.join(file_dir, self.results_save_name)

        np.savez(
            file_path,
            actions=self.agent_action_history,
            env_status=self.agent_env_status_history,
            peer_pressure_coeff=self.agent_peer_pressure_coeff_history,
            env_perception_coeff=self.agent_env_perception_coeff_history,
            env_utility=self.agent_env_utility_history,
        )

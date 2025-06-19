"""Agent-based model base class for simulations.

This module defines a base class for agent-based models, providing
a framework for initializing agents, calculating neighbor actions,
and managing the simulation environment.

"""

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

    def __init__(
        self,
        num_agents: int = DEFAULT_NUM_AGENTS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        radius: int = DEFAULT_RADIUS,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
        adaptive_attr_option: str = DEFAULT_ADAPTIVE_ATTR_OPTION,
        rng: np.random.Generator = None,
        env_status_fn=None,
        peer_pressure_coeff_fn=None,
        env_perception_coeff_fn=None,
    ):
        """Initialize the base model with a grid of agents.

        Args:
            num_agents (int): Total number of agents in the grid.
            width (int): Width of the grid.
            height (int): Height of the grid.
            radius (int): Radius for neighbor calculations.
            memory_count (int): Number of past actions to remember for each agent.
            env_update_option (str, optional):
                Method to update the environment status.
            adaptive_attr_option (str, optional):
                Option for adaptive attributes. Defaults to None.
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
        self.num_agents = num_agents
        self.radius = radius
        self.width = width
        self.height = height
        self.memory_count = memory_count
        self.env_update_option = env_update_option
        self.adaptive_attr_option = adaptive_attr_option
        self.rng = rng or np.random.default_rng()

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
                    env_status_fn,
                    peer_pressure_coeff_fn,
                    env_perception_coeff_fn,
                )
                i += 1

        self.agent_action_history = [self.get_agent_actions()]
        self.agent_env_status_history = [self.get_agent_env_status()]
        self.agent_peer_pressure_coeff_history = [self.get_agent_peer_pressure_coeff()]
        self.agent_env_utility_history = [self.get_agent_env_utility()]

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
                all_peer_actions = self.get_neighbor_actions(x, y)
                all_peer_env_utilities = self.get_neighbor_env_utilities(x, y)
                agent.decide_action(
                    ave_peer_action, all_peer_actions, all_peer_env_utilities
                )

    def run(self, steps: int = 20) -> None:
        """Run the model for a specified number of steps.

        Args:
            steps (int): Number of steps to run the model.
        """
        for _ in range(steps):
            self.step()
            self.agent_action_history.append(self.get_agent_actions())
            self.agent_env_status_history.append(self.get_agent_env_status())
            self.agent_peer_pressure_coeff_history.append(
                self.get_agent_peer_pressure_coeff()
            )
            self.agent_env_utility_history.append(self.get_agent_env_utility())

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

    def get_neighbor_actions(self, x: int, y: int) -> np.ndarray:
        """Get the actions of neighboring agents.

        This method retrieves the most recent actions of all
        neighbors of the agent at position (x, y).

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            np.ndarray: Array of actions from neighboring agents.
        """
        neighbor_actions = []
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor.past_actions:
                neighbor_actions.append(neighbor.past_actions[-1])
            else:
                neighbor_actions.append(0)
        return np.array(neighbor_actions)

    def get_neighbor_env_utilities(self, x: int, y: int) -> np.ndarray:
        """Get the environment utilities of neighboring agents.

        This method retrieves the most recent environment utilities
        of all neighbors of the agent at position (x, y).

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            np.ndarray: Array of environment utilities from neighboring agents.
        """
        neighbor_env_utilities = []
        neighbors = self.get_neighbors(x, y)
        for neighbor in neighbors:
            if neighbor.env_utility_history:
                neighbor_env_utilities.append(neighbor.env_utility_history[-1])
            else:
                neighbor_env_utilities.append(0)
        return np.array(neighbor_env_utilities)

    def get_agent_actions(self) -> np.ndarray:
        """Get the actions of all agents in a 2D array.

        Returns:
            np.ndarray: A 2D array where each cell contains the
            action of the agent at that position.
        """
        action_grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                action_grid[x, y] = self.agents[x, y].get_recent_action()
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
                env_status_grid[x, y] = self.agents[x, y].get_recent_env_status()
        return env_status_grid

    def get_agent_peer_pressure_coeff(self) -> np.ndarray:
        """Get the peer pressure coefficients of all agents in a 2D array.

        Returns:
            np.ndarray: A 2D array where each cell contains the
            peer pressure coefficient of the agent at that position.
        """
        peer_pressure_grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                peer_pressure_grid[x, y] = self.agents[
                    x, y
                ].get_recent_peer_pressure_coeff()
        return peer_pressure_grid

    def get_agent_env_utility(self) -> np.ndarray:
        """Get the environment utility of all agents in a 2D array.

        Returns:
            np.ndarray: A 2D array where each cell contains the
            environment utility of the agent at that position.
        """
        env_utility_grid = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                env_utility_grid[x, y] = self.agents[x, y].get_recent_env_utility()
        return env_utility_grid

    def get_agent_actions_at_time(self, time: int) -> np.ndarray:
        """Get the actions of all agents at a specific time step.

        Args:
            time (int): The time step to retrieve actions for.

        Returns:
            np.ndarray: A 2D array of agent actions at the specified time.
        """
        if time < len(self.agent_action_history):
            return self.agent_action_history[time]
        else:
            raise IndexError("Time step exceeds the history length.")

    def get_agent_env_status_at_time(self, time: int) -> np.ndarray:
        """Get the environment status of all agents at a specific time step.

        Args:
            time (int): The time step to retrieve environment status for.

        Returns:
            np.ndarray: A 2D array of agent environment status at the specified time.
        """
        if time < len(self.agent_env_status_history):
            return self.agent_env_status_history[time]
        else:
            raise IndexError("Time step exceeds the history length.")

    def get_agent_peer_pressure_coeff_at_time(self, time: int) -> np.ndarray:
        """Get the peer pressure coefficients of all agents at a specific time step.

        Args:
            time (int): The time step to retrieve peer pressure coefficients for.

        Returns:
            np.ndarray: A 2D array of agent peer pressure
                coefficients at the specified time.
        """
        if time < len(self.agent_peer_pressure_coeff_history):
            return self.agent_peer_pressure_coeff_history[time]
        else:
            raise IndexError("Time step exceeds the history length.")

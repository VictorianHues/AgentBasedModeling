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
        forecast_threshold (float):
            Threshold for forecasting agent actions.
    """

    DEFAULT_NUM_AGENTS = 100
    DEFAULT_WIDTH = 10
    DEFAULT_HEIGHT = 10
    DEFAULT_RADIUS = 1
    DEFAULT_MEMORY_COUNT = 1
    DEFAULT_ENV_UPDATE_OPTION = "linear"
    DEFAULT_FORECAST_THRESHOLD = 0.6

    def __init__(
        self,
        num_agents: int = DEFAULT_NUM_AGENTS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        radius: int = DEFAULT_RADIUS,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        forecast_threshold: float = DEFAULT_FORECAST_THRESHOLD,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
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
            rng (np.random.Generator, optional):
                Random number generator. Defaults to None.
            env_update_option (str, optional):
                Method to update the environment status.
            env_status_fn (callable, optional):
                Function to initialize env_status.
            peer_pressure_coeff_fn (callable, optional):
                Function to initialize peer_pressure_coeff.
            env_perception_coeff_fn (callable, optional):
                Function to initialize env_perception_coeff.
            forecast_threshold (float, optional):
                Threshold for forecasting agent actions. Defaults to 0.6.
        """
        self.time = 0
        self.num_agents = num_agents
        self.radius = radius
        self.width = width
        self.height = height
        self.memory_count = memory_count
        self.env_update_option = env_update_option.lower()
        self.rng = rng or np.random.default_rng()
        self.forecast_threshold = forecast_threshold

        self.agents = np.empty((width, height), dtype=object)
        i = 0
        for x in range(self.width):
            for y in range(self.height):
                self.agents[x, y] = Agent(
                    i,
                    self.memory_count,
                    self.rng,
                    self.env_update_option,
                    env_status_fn,
                    peer_pressure_coeff_fn,
                    env_perception_coeff_fn,
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
                total_action += np.mean(neighbor.pdast_actions)
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
                env_status_grid[x, y] = self.agents[x, y].env_status[-1]
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

    # def step(self):
    #     # increment time
    #     self.time += 1

    #     # compute last-step coop fraction for score updates
    #     if len(self.agent_action_history) > 0:
    #         last_actions = self.agent_action_history[-1]
    #         prev_coop_frac = (last_actions == +1).sum() / last_actions.size
    #     else:
    #         prev_coop_frac = 0.5  # or whatever your default is

    #     # have each agent decide (pass history of coop fractions)
    #     coop_history = [(h == +1).sum() / h.size for h in self.agent_action_history]
    #     for x in range(self.width):
    #         for y in range(self.height):
    #             agent = self.agents[x, y]
    #             neighb_avg = self.ave_neighb_action_single_memory(x, y)
    #             agent.decide_action(neighb_avg, coop_history)

    #     # now update predictor scores using the previous actual
    #     for x in range(self.width):
    #         for y in range(self.height):
    #             agent = self.agents[x, y]
    #             agent.update_predictor_scores(prev_coop_frac)

    #     # finally record histories as usual
    #     self.agent_action_history.append(self.get_agent_actions())
    #     self.agent_env_status_history.append(self.get_agent_env_status())

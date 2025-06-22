"""Agent-based model base class for simulations.

This module defines a base class for agent-based models, providing
a framework for initializing agents, calculating neighbor actions,
and managing the simulation environment.

"""

import concurrent.futures
import os

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

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
    DEFAULT_PREDICTION_OPTION = "linear"
    DEFAULT_ENV_UPDATE_OPTION = "linear"
    DEFAULT_ADAPTIVE_ATTR_OPTION = None
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_RATIONALITY = 1.0

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        radius: int = DEFAULT_RADIUS,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
        adaptive_attr_option: str = DEFAULT_ADAPTIVE_ATTR_OPTION,
        neighb_prediction_option: str = DEFAULT_PREDICTION_OPTION,
        peer_pressure_learning_rate: float = DEFAULT_LEARNING_RATE,
        rationality: float = DEFAULT_RATIONALITY,
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
            neighb_prediction_option (str, optional):
                Method for neighbor action prediction.
            peer_pressure_learning_rate (float, optional):
                Learning rate for peer pressure updates.
            rationality (float, optional):
                Rationality factor for agent decisions.
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
        self.neighb_prediction_option = neighb_prediction_option
        self.peer_pressure_learning_rate = peer_pressure_learning_rate
        self.rationality = rationality

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
                    self.peer_pressure_learning_rate,
                    self.rationality,
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
        agents to decide their actions in parallel.
        """
        self.time += 1

        def process_agent(args):
            x, y = args
            agent = self.agents[x, y]
            ave_peer_action = self.pred_neighb_action(x, y)
            all_peer_actions = self.get_neighbor_attribute_values(x, y, "past_actions")
            agent.decide_action(ave_peer_action, all_peer_actions)

        positions = [(x, y) for x in range(self.width) for y in range(self.height)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(process_agent, positions))

    def run(self, steps: int = 20) -> None:
        """Run the model for a specified number of steps.

        Args:
            steps (int): Number of steps to run the model.
        """
        for _ in tqdm(range(steps)):
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

    @staticmethod
    def sigmoid(x, a, b):
        """Sigmoid function for logistic regression.

        This function defines a sigmoid curve for logistic regression
        fitting, which maps any real-valued number into the range [0, 1].

        Args:
            x (float or np.ndarray): Input value(s) to the sigmoid function.
            a (float): Slope of the sigmoid curve.
            b (float): Offset of the sigmoid curve.

        Returns:
            float or np.ndarray: Sigmoid-transformed value(s).
        """
        return 1 / (1 + np.exp(-(a * x + b)))

    def pred_neighb_action(self, x: int, y: int) -> float:
        """Predict the average action of peers based on their recent actions.

        This method predicts the average action of neighboring agents
        based on their most recent actions, using linear regression.

        Args:
            x (int): X-coordinate of the agent.
            y (int): Y-coordinate of the agent.

        Returns:
            float: Predicted average action of neighbors.
        """
        neighbors = self.get_neighbors(x, y)
        predicted_actions = []
        total_predicted_action = 0

        if self.neighb_prediction_option == "linear":
            for neighbor in neighbors:
                if len(neighbor.past_actions) < 2:
                    return self.ave_neighb_action(x, y, self.memory_count)
                else:
                    # Fit a linear regression model to the past actions
                    actions = np.array(neighbor.past_actions)
                    time_steps = np.arange(len(actions))
                    coeffs = np.polyfit(time_steps, actions, 1)
                    # Predict the next action based on the last time step
                    predicted_action = np.polyval(coeffs, len(actions))
                    predicted_actions.append(predicted_action)
            # Return the mean of the predicted actions
            if predicted_actions:
                mean_predicted_action = np.mean(predicted_actions)
                if mean_predicted_action >= 0:
                    total_predicted_action = 1
                elif mean_predicted_action < 0:
                    total_predicted_action = -1
        elif self.neighb_prediction_option == "logistic":
            neighbors = self.get_neighbors(x, y)
            predicted_probs = []
            total_predicted_action = 0

            for neighbor in neighbors:
                if len(neighbor.past_actions) < 2:
                    return self.ave_neighb_action(x, y, self.memory_count)
                else:
                    actions = np.array(neighbor.past_actions)

                    # If actions are -1/1, map to 0/1 for logistic regression
                    if set(actions) <= {-1, 1}:
                        actions = (actions + 1) // 2
                    time_steps = np.arange(len(actions))
                    log_time = np.log(time_steps + 1)
                    try:
                        popt, _ = curve_fit(
                            self.sigmoid, log_time, actions, maxfev=10000
                        )

                        pred_prob = self.sigmoid(np.log(len(actions) + 1), *popt)
                    except Exception:
                        pred_prob = np.mean(actions)
                    predicted_probs.append(pred_prob)
            if predicted_probs:
                mean_prob = np.mean(predicted_probs)
                total_predicted_action = 1 if mean_prob >= 0.5 else -1
            else:
                total_predicted_action = self.ave_neighb_action(x, y, self.memory_count)
        else:
            print(
                "No predicted actions available, "
                "using average of neighbors' last actions."
            )
            total_predicted_action = self.ave_neighb_action(x, y, self.memory_count)

        return total_predicted_action

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
        """Get a 2D grid of a specific agent attribute.

        This method retrieves the most recent values of a specified
        attribute from all agents in the grid and returns it as a 2D array.

        Args:
            attribute (str): The attribute to
            retrieve from agents (e.g., 'past_actions').

        Returns:
            np.ndarray: A 2D array of the specified attribute values across the grid.
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

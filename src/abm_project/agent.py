"""Agent class for an agent-based model simulation.

This class represents an agent that interacts with its environment and peers.
It includes methods for decision-making based on peer actions and environmental status.

"""

import numpy as np


class Agent:
    """Agent class for an agent-based model simulation.

    This class represents an agent that interacts
    with its environment and peers.
    It includes methods for decision-making based on
    peer actions and environmental status.
    """

    ACTIONS = [-1, 1]
    DEFAULT_MEMORY_COUNT = 1
    DEFAULT_ENV_UPDATE_OPTION = "linear"

    def __init__(
        self,
        id: int,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        rng: np.random.Generator = None,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
        env_status_fn=None,
        peer_pressure_coeff_fn=None,
        env_perception_coeff_fn=None,
        forecast_threshold: float = 0.6,  # comfort level
    ):
        """Initialize an agent.

        Initialize with an ID, environment status,
        peer pressure coefficient, and environment
        perception coefficient.

        Args:
            id (int):
                Unique identifier for the agent.
            memory_count (int):
                Number of past steps to remember.
            rng (np.random.Generator, optional):
                Random number generator. Defaults to None.
            env_update_option (str):
                Method to update the environment status.
            env_status_fn (callable):
                Function that returns the initial status of the environment,
                typically between -1 and 1.
            peer_pressure_coeff_fn (callable):
                Function that returns the peer pressure coefficient.
            env_perception_coeff_fn (callable):
                Function that returns the agent's perception coefficient
                of the environment.
            forecast_threshold (float):
                Threshold for deciding the action based on forecasts.
                If the forecast is below this threshold, the agent will take action 1,
                otherwise it will take action -1.
        """
        self.id = id
        self.memory_count = memory_count
        self.rng = rng or np.random.default_rng()
        self.env_update_option = env_update_option
        self.forecast_threshold = forecast_threshold

        # define predictors
        # TODO: move to function outsode init())
        def last(history):
            return history[-1] if len(history) > 0 else 0.5

        def linear(history):
            """Predict the next value using linear regression."""
            if not history:
                return 0.5
            # If history is too short, return the last value
            if len(history) < 2:
                return last(history)
            x = np.arange(len(history))
            y = np.array(history)
            # fit y = a*x + b
            a, b = np.polyfit(x, y, 1)
            return a * len(history) + b

        self.predictors = {
            "last": last,
            "linear": linear,
        }

        # storage for the last round of forecasts
        # TODO: move to function outsode init())
        self._last_forecasts = {}

        self.past_actions = [
            self.rng.choice(self.ACTIONS) for _ in range(self.memory_count)
        ]
        # Only have a randomly generated strting action...(why randomly generated?)\
        self.env_status = [env_status_fn() for _ in range(self.memory_count)]
        self.peer_pressure_coeff = [
            peer_pressure_coeff_fn() for _ in range(self.memory_count)
        ]
        self.env_perception_coeff = [
            env_perception_coeff_fn() for _ in range(self.memory_count)
        ]

    def update_environment_status(self, action_decision: int) -> None:
        """Update the environment status based on the agent's action.

        The environment status is updated based on the agent's action
        and the current environment status. The update is done using
        a sigmoid function, exponential decay, or linear update,
        depending on the `env_update_option` specified during initialization.
        The formula for the update is:
        env_status(t+1) = env_status(t) + delta
        where delta is calculated based on the action decision
        and the current environment status.

        Args:
            action_decision (int): The action taken by the agent, either -1 or 1.

        Raises:
            ValueError: If the `env_update_option` is invalid.
        """
        current_env_status = self.get_recent_env_status()
        if self.env_update_option == "sigmoid":
            sensitivity = 1 / (1 + np.exp(6 * (current_env_status - 0.5)))
            delta = sensitivity * action_decision * 0.05
        elif self.env_update_option == "exponential":
            delta = action_decision * 0.05 * np.exp(-current_env_status)
        elif self.env_update_option == "linear":
            delta = action_decision * 0.05
        else:
            raise ValueError("Invalid environment update option.")

        current_env_status += delta
        current_env_status = max(0.0, min(1, current_env_status))
        self.env_status.append(current_env_status)
        if len(self.env_status) > self.memory_count:
            self.env_status.pop(0)

    def calculate_deviation_cost(self, action: int, ave_peer_action: float) -> float:
        """Calculate the cost of deviating from the average peer action.

        The cost is calculated as:
        c * (a_i(t) - A_i(t))^2

        Args:
            action (int): The action taken by the agent, either -1 or 1.
            ave_peer_action (float): The average action of peers.

        Returns:
            float: The cost of deviation from your neighbors.
        """
        return self.peer_pressure_coeff[-1] * (action - ave_peer_action) ** 2

    def calculate_perceived_severity(self) -> float:
        """Calculate the perceived severity of the environment.

        The perceived severity is a function of the environment
        status and the agent's perception coefficient.
        It is calculated as:
        env_perception_coeff * env_status * -1
        The negative sign indicates that a higher environment status
        leads to a lower perceived severity.
        The perceived severity is used to determine the utility of actions.

        Returns:
            float: The perceived severity of the environment.
        """
        return self.env_perception_coeff[-1] * (2 * self.env_status[-1] - 1) * -1.0

    def calculate_action_utility(self, action: int, ave_peer_action: float) -> float:
        """Calculate the utility of taking a specific action.

        The utility is calculated as the perceived severity
        of the environment multiplied by the action, minus
        the cost of deviating from the average peer action.
        The formula is:
        V_i(a_i(t)) = a_i(t) * U_i(t) - c * (a_i(t) - A_i(t))^2

        Args:
            action (int): The action taken by the agent, either -1 or 1.
            ave_peer_action (float): The average action of peers.

        Returns:
            float: The utility of the action.
        """
        deviation_cost = self.calculate_deviation_cost(action, ave_peer_action)
        perceived_severity = self.calculate_perceived_severity()
        env_action_utility = action * perceived_severity
        return env_action_utility - deviation_cost

    def calculate_action_probabilities(self, ave_peer_action: float) -> np.ndarray:
        """Calculate the probability of each possible action.

        The probabilities are calculated using a logit softmax
        function over the utilities of each action.
        The formula is:
        P(a_i(t) = a) = exp(V_i(a)) / sum(exp(V_i(a')) for all a')
        where V_i(a) is the utility of action a for agent i.

        Args:
            ave_peer_action (float): The average action of peers.

        Returns:
            np.ndarray: An array of probabilities for each action.
        """
        utilities = np.array(
            [self.calculate_action_utility(a, ave_peer_action) for a in self.ACTIONS]
        )
        exp_utilities = np.exp(utilities - np.max(utilities))
        probabilities = exp_utilities / np.sum(exp_utilities)
        return probabilities

    def update_past_actions(self, action) -> None:
        """Update the memory of past actions.

        This method maintains a fixed-length memory of past actions.
        If the memory exceeds the specified count, the oldest action is removed.
        """
        if len(self.past_actions) >= self.memory_count:
            self.past_actions.pop(0)
        self.past_actions.append(action)

    def decide_action(
        self, ave_peer_action: float, global_coop_history: list[float]
    ) -> None:
        """Decide on a new action based on peer actions and environment."""
        forecasts = {
            name: fn(global_coop_history) for name, fn in self.predictors.items()
        }
        best = max(self.predictor_scores, key=lambda name: self.predictor_scores[name])
        forecast = forecasts[best]
        action = 1 if forecast < self.forecast_threshold else -1

        self.update_past_actions(action=action)
        self.update_environment_status(action_decision=action)

        # Store the last forecast for potential future use
        self._last_forecasts[self.id] = forecast

        # probabilities = self.calculate_action_probabilities(ave_peer_action)
        # action = self.rng.choice(self.ACTIONS, p=probabilities)
        # self.update_past_actions(action)
        # self.update_environment_status(action)

    def get_recent_action(self) -> int:
        """Get the most recent action taken by the agent."""
        return self.past_actions[-1] if self.past_actions else None

    def get_recent_env_status(self) -> float:
        """Get the most recent environment status perceived by the agent."""
        return self.env_status[-1] if self.env_status else None

    def get_recent_peer_pressure_coeff(self) -> float:
        """Get the most recent peer pressure coefficient perceived by the agent."""
        return self.peer_pressure_coeff[-1] if self.peer_pressure_coeff else None

    def get_recent_env_perception_coeff(self) -> float:
        """Get the most recent environment perception coefficient of the the agent."""
        return self.env_perception_coeff[-1] if self.env_perception_coeff else None

    def get_last_forecast(self) -> float:
        """Get the last forecast made by the agent."""
        return self._last_forecasts.get(self.id, None)

    def update_predictor_scores(self, actual_coop_frac: float) -> None:
        """Update the scores of the predictors based on their performance.

        After actual cooperation fraction is known, penalize by abs error.
        """
        for name, pred in self._last_forecasts.items():
            error = abs(actual_coop_frac - pred)
            # you can also do self.predictor_scores[name] += (1 - error)
            self.predictor_scores[name] -= error

        return self.predictor_scores

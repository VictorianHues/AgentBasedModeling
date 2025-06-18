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
    HISTORY_LENGTH = 10

    def __init__(
        self,
        id: int,
        env_status: float,
        peer_pressure_coeff: float,
        env_perception_coeff: float,
        rng: np.random.Generator = None,
    ):
        """Initialize an agent.

        Initialize with an ID, environment status,
        peer pressure coefficient, and environment
        perception coefficient.

        Args:
            id (int):
                Unique identifier for the agent.
            env_status (float):
                Initial status of the environment, typically between -1 and 1.
            peer_pressure_coeff (float):
                Coefficient representing the influence of peer pressure.
            env_perception_coeff (float):
                Coefficient representing the agent's perception of the environment.
            rng (np.random.Generator, optional):
                Random number generator. Defaults to None.
        """
        self.id = id
        self.env_status = env_status
        self.peer_pressure_coeff = peer_pressure_coeff
        self.env_perception_coeff = env_perception_coeff
        self.rng = rng or np.random.default_rng()
        self.action = self.rng.choice(self.ACTIONS)
        self.past_actions = [
            self.rng.choice(self.ACTIONS) for _ in range(self.HISTORY_LENGTH)
        ]

    def update_env_perception_coeff(self) -> float:
        """Update the agent's perception coefficient of the environment.

        This coefficient is used to calculate the perceived
        severity of the environment.
        """
        return self.env_perception_coeff

    def update_peer_pressure_coeff(self) -> float:
        """Update the agent's peer pressure coefficient.

        This coefficient is used to calculate the cost
        of deviating from the average peer action.
        """
        return self.peer_pressure_coeff

    def update_environment_status(self, action_decision: int) -> None:
        """Update the agent's perception of the environment status.

        The environment status is updated based on the agent's action decision.
        The status is limited to the range [-1, 1].

        Args:
            action_decision (int): The action taken by the agent, either -1 or 1.
        """
        self.env_status += 0.1 * action_decision
        self.env_status = max(-1, min(1, self.env_status))

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
        return self.peer_pressure_coeff * (action - ave_peer_action) ** 2

    def calculate_perceived_severity(self) -> float:
        """Calculate the perceived severity of the environment.

        The perceived severity is a function of the environment
        status and the agent's perception coefficient.
        It is calculated as:
        env_perception_coeff * env_status

        Returns:
            float: The perceived severity of the environment.
        """
        return self.env_perception_coeff * self.env_status * -1.0

    def calculate_action_utility(self, action: int, ave_peer_action: float) -> float:
        """Calculate the utility of taking a specific action.

        The utility is calculated as the perceived severity
        of the environment multiplied by the action, minus
        the cost of deviating from the average peer action.
        The formula is:
        V_i(a_i(t)) = U_i(t) - c * (a_i(t) - A_i(t))^2

        Args:
            action (int): The action taken by the agent, either -1 or 1.
            ave_peer_action (float): The average action of peers.

        Returns:
            float: The utility of the action.
        """
        deviation_cost = self.calculate_deviation_cost(action, ave_peer_action)
        perceived_severity = self.calculate_perceived_severity()
        env_action_utility = perceived_severity * action
        return env_action_utility - deviation_cost

    def calculate_action_probabilities(
        self, ave_peer_action: float
    ) -> tuple[list[int], np.ndarray]:
        """Calculate the probability of each possible action.

        The probabilities are calculated using a logit softmax
        function over the utilities of each action.
        The formula is:
        P(a_i(t) = a) = exp(V_i(a)) / sum(exp(V_i(a')) for all a')
        where V_i(a) is the utility of action a for agent i.

        Args:
            ave_peer_action (float): The average action of peers.

        Returns:
            Tuple[List[int], np.ndarray]:
                A tuple containing the list of actions and
                their corresponding probabilities.
        """
        utilities = np.array(
            [self.calculate_action_utility(a, ave_peer_action) for a in self.ACTIONS]
        )
        exp_utilities = np.exp(utilities - np.max(utilities))  # for numerical stability
        probabilities = exp_utilities / np.sum(exp_utilities)
        return self.ACTIONS, probabilities

    def update_past_actions(self) -> None:
        """Update the history of past actions.

        This method maintains a fixed-length history of past actions.
        If the history exceeds the defined length, the oldest action is removed.
        """
        if len(self.past_actions) >= self.HISTORY_LENGTH:
            self.past_actions.pop(0)
        self.past_actions.append(self.action)

    def decide_action(self, ave_peer_action: float) -> None:
        """Decide on a new action based on peer actions and environment."""
        action_set, probabilities = self.calculate_action_probabilities(ave_peer_action)
        self.action = self.rng.choice(action_set, p=probabilities)
        self.update_past_actions()
        self.update_environment_status(self.action)

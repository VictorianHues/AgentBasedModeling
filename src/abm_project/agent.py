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
    DEFAULT_ADAPTIVE_ATTR_OPTION = None

    def __init__(
        self,
        id: int,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        rng: np.random.Generator = None,
        env_update_option: str = DEFAULT_ENV_UPDATE_OPTION,
        adaptive_attr_option: str = DEFAULT_ADAPTIVE_ATTR_OPTION,
        env_perception_learning_rate=0.1,
        peer_pressure_learning_rate=0.2,
        env_status_fn=None,
        peer_pressure_coeff_fn=None,
        env_perception_coeff_fn=None,
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
            adaptive_attr_option (str):
                Method to adapt the agent's attributes.
            env_perception_learning_rate (float):
                Learning rate for the agent's perception of the environment.
            peer_pressure_learning_rate (float):
                Learning rate for the agent's peer pressure coefficient.
            env_status_fn (callable):
                Function that returns the initial status of the environment,
                typically between -1 and 1.
            peer_pressure_coeff_fn (callable):
                Function that returns the peer pressure coefficient.
            env_perception_coeff_fn (callable):
                Function that returns the agent's perception coefficient
                of the environment.
        """
        self.id = id
        self.memory_count = memory_count
        self.rng = rng or np.random.default_rng()
        self.env_update_option = env_update_option
        self.adaptive_attr_option = adaptive_attr_option
        self.env_perception_learning_rate = env_perception_learning_rate
        self.peer_pressure_learning_rate = peer_pressure_learning_rate

        self.past_actions = [
            self.rng.choice(self.ACTIONS) for _ in range(self.memory_count)
        ]
        self.env_status = [env_status_fn() for _ in range(self.memory_count)]
        self.peer_pressure_coeff = [
            peer_pressure_coeff_fn() for _ in range(self.memory_count)
        ]
        self.env_perception_coeff = [
            env_perception_coeff_fn() for _ in range(self.memory_count)
        ]

        self.env_utility_history = []
        self.deviation_pressure_cost = []

    def decide_action(
        self, ave_peer_action: float, all_peer_actions: np.ndarray
    ) -> None:
        """Decide on a new action based on peer actions and environment."""
        probabilities = self.calculate_action_probabilities(ave_peer_action)
        action = self.rng.choice(self.ACTIONS, p=probabilities)
        self.update_past_actions(action)
        self.update_environment_status(action)
        self.update_peer_pressure_coeff(all_peer_actions)
        self.update_env_perception_coeff()

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

    def update_past_actions(self, action: int) -> None:
        """Update the memory of past actions.

        This method maintains a fixed-length memory of past actions.
        If the memory exceeds the specified count, the oldest action is removed.
        """
        if len(self.past_actions) >= self.memory_count:
            self.past_actions.pop(0)
        self.past_actions.append(action)

    def update_env_perception_coeff(self) -> float:
        """Update the agent's environment perception coefficient.

        This coefficient is used to calculate the agent's perception
        of the environment's status. The update is done using a sigmoid function
        that considers the agent's last action and the current environment status.
        The formula for the update is:
        new_coeff = (1 - learning_rate) * old_coeff + learning_rate * sigmoid_s
        ensitivity
        where sigmoid_sensitivity is calculated based on the agent's last action
        and the current environment status.
        The sigmoid sensitivity is defined as:
        sigmoid_sensitivity = 1 / (1 + exp(past_action * k * (env_status - 0.5)))
        where k is a steepness parameter that can be adjusted.
        """
        if self.adaptive_attr_option == "bayesian":
            learning_rate = self.env_perception_learning_rate
            k = 8
            env_status = self.env_status[-1]

            sigmoid_sensitivity = 1 / (
                1 + np.exp(self.past_actions[-1] * k * (env_status - 0.5))
            )

            new_coeff = (
                (1 - learning_rate) * self.env_perception_coeff[-1]
                + learning_rate * sigmoid_sensitivity  # env_status
            )
            # print(f"Env Status: {env_status}, Action: {self.past_actions[-1]},
            # Sigmoid Sense: {sigmoid_sensitivity}, New Coeff: {new_coeff},
            # old Coeff: {self.env_perception_coeff[-1]}")
        else:
            new_coeff = self.env_perception_coeff[-1]

        self.env_perception_coeff.append(new_coeff)
        if len(self.env_perception_coeff) > self.memory_count:
            self.env_perception_coeff.pop(0)

    def update_peer_pressure_coeff(self, all_peer_actions: np.ndarray) -> float:
        """Update the agent's peer pressure coefficient.

        This coefficient is used to calculate the cost
        of deviating from the average peer action.
        """
        if self.adaptive_attr_option == "bayesian":
            learning_rate = self.peer_pressure_learning_rate
            # k = 10  # steepness of sigmoid curve

            proportion_action_pos = np.sum(all_peer_actions == 1) / len(
                all_peer_actions
            )
            proportion_action_neg = np.sum(all_peer_actions == -1) / len(
                all_peer_actions
            )

            consensus = max(proportion_action_pos, proportion_action_neg)
            majority_action = (
                1 if proportion_action_pos >= proportion_action_neg else -1
            )

            norm_consensus = (consensus - 0.5) * 2  # -1 to 1
            confidence = norm_consensus  # 1 / (1 + np.exp(-k * norm_consensus))

            confidence = 2 * (confidence - 0.5)  # [-1, 1]

            # Add slight noise to prevent perfect convergence
            # confidence += np.random.normal(0, 0.05)

            # Flip update if recently agreed but majority flipped (or vice versa)
            past_action = self.past_actions[-1]
            agreed_with_majority = past_action == majority_action
            previous_majority = (
                self.previous_majority
                if hasattr(self, "previous_majority")
                else majority_action
            )

            majority_flipped = majority_action != previous_majority
            if majority_flipped:
                confidence *= -1

            if agreed_with_majority:
                new_coeff = self.peer_pressure_coeff[-1] + learning_rate * confidence
            else:
                new_coeff = self.peer_pressure_coeff[-1] - learning_rate * confidence

            new_coeff = np.clip(new_coeff, 0, 1)

            self.previous_majority = majority_action

            # print(f"Peer Pressure Coeff Update: {self.peer_pressure_coeff[-1]} ->
            # {new_coeff}, Action: {self.past_actions[-1]},
            # Majority Action: {majority_action},
            # Confidence: {confidence},
            # Proportion Pos: {proportion_action_pos},
            # Proportion Neg: {proportion_action_neg}")

            new_coeff = np.clip(new_coeff, 0, 1)
        else:
            new_coeff = self.peer_pressure_coeff[-1]

        self.peer_pressure_coeff.append(new_coeff)
        if len(self.peer_pressure_coeff) > self.memory_count:
            self.peer_pressure_coeff.pop(0)

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

        Sigmoid: Rate of change is higher when the environment status is around 0.5.
            Lowest delta is at 1, highest delta is at 0.0.
        Sigmoid Asymmetric: Delta is asymmetric based on the action decision.
            Positive delta is lower when the environment status is low,
            and negative delta is higher when the environment status is low.
        Exponential: Rate of change decreases as the environment status increases.
            Lowest delta is at 1, highest delta is at 0.0.
        Linear: Delta is a constant value based on the action decision.
        Bell: Lowest delta at 0 and 1, highest delta at 0.5.
        Bimodal: Highest delta at two peaks, around 0.25 and 0.75,
        and lowest delta at 0, 0.5, and 1.

        Args:
            action_decision (int): The action taken by the agent, either -1 or 1.

        Raises:
            ValueError: If the `env_update_option` is invalid.
        """
        current_env_status = self.env_status[-1]
        if self.env_update_option == "sigmoid":
            sensitivity = 1 / (1 + np.exp(8 * (current_env_status - 0.5)))
            delta = sensitivity * action_decision * 0.05
        elif self.env_update_option == "sigmoid_asymmetric":
            exponent = -action_decision * 8 * (current_env_status - 0.5)
            denominator = 1 + np.exp(exponent)
            sensitivity = 1 / denominator
            delta = sensitivity * action_decision * 0.05
        elif self.env_update_option == "exponential":
            delta = action_decision * 0.05 * np.exp(-current_env_status)
        elif self.env_update_option == "linear":
            delta = action_decision * 0.05
        elif self.env_update_option == "bell":
            delta = (
                action_decision * 0.2 * current_env_status * (1 - current_env_status)
            )
        elif self.env_update_option == "bimodal":
            min_update = 0.01
            max_update = 0.05
            delta = (
                action_decision
                * (max_update - min_update)
                * np.sin(2 * np.pi * current_env_status)
                + min_update
            )
        else:
            raise ValueError("Invalid environment update option.")

        current_env_status += delta
        current_env_status = max(0.01, min(0.99, current_env_status))
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

        self.env_utility_history.append(env_action_utility)
        if len(self.env_utility_history) > self.memory_count:
            self.env_utility_history.pop(0)
        self.deviation_pressure_cost.append(deviation_cost)
        if len(self.deviation_pressure_cost) > self.memory_count:
            self.deviation_pressure_cost.pop(0)

        return env_action_utility - deviation_cost

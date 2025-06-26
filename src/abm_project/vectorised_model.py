"""Vectorised implementation of the OOP model."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy

from .oop_model import BaseModel
from .utils import lattice2d, piecewise_exponential_update

type EnvUpdateFn = Callable[(npt.NDArray[float], npt.NDArray[int]), npt.NDArray[float]]


class VectorisedModel:
    """Vectorised base model for agent-based simulations.

    This model comprises a 2D lattice of agents who repeatedly choose between
    cooperation (pro-environmental behaviour) and defection, based on the state
    of their local environment and the social norms imposed by their direct
    neighbors.

    Agents have heterogeneous attributes which weight the respective contributions
    of environmental concern and social norms in the decision-making process.

    Attributes:
        action (npt.NDArray[np.int64]):
            2D array of agents' actions with shape (time, agent).
        environment (npt.NDArray[np.int64]):
            2D array of agents' environments with shape (time, agent).
        s (npt.NDArray[np.float64]):
            2D array of agents' support for cooperation with shape (time, agent).
        b (npt.NDArray[np.float64]):
            2D array of agents' decision-making weights, shape (attributes, agent).
        rationality: float
            Homogeneous rationality coefficient for all agents.
        adj (npt.NDArray[np.float64]):
            Normalised adjacency matrix with shape (agent, agent).
        time (int): Current time step in the simulation.
        num_agents (int): Total number of agents in the grid.
        width (int): Width of the grid.
        height (int): Height of the grid.
        simmer_time (int): Number of agent adaptation steps between environment updates.
        rng (np.random.Generator):
            Random number generator for stochastic processes.

    """

    DEFAULT_NUM_AGENTS = BaseModel.DEFAULT_NUM_AGENTS
    DEFAULT_WIDTH = BaseModel.DEFAULT_WIDTH
    DEFAULT_HEIGHT = BaseModel.DEFAULT_HEIGHT
    DEFAULT_MEMORY_COUNT = BaseModel.DEFAULT_MEMORY_COUNT
    DEFAULT_MAX_STORAGE = 1000
    DEFAULT_SIMMER_TIME = 1
    DEFAULT_NEIGHB_PREDICTION_OPTION = "linear"  # "logistic", None
    DEFAULT_SEVERITY_BENEFIT_OPTION = "adaptive"  # None
    DEFAULT_RADIUS_OPTION = "single"  # "all"

    ACTIONS = [-1, 1]
    N_WEIGHTS = 2

    def __init__(
        self,
        num_agents: int = DEFAULT_NUM_AGENTS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        memory_count: int = DEFAULT_MEMORY_COUNT,
        env_update_fn: EnvUpdateFn | None = None,
        rng: np.random.Generator = None,
        rationality: float = 1.0,
        max_storage: int = DEFAULT_MAX_STORAGE,
        moore: bool = True,
        simmer_time: int = DEFAULT_SIMMER_TIME,
        neighb_prediction_option: str = DEFAULT_NEIGHB_PREDICTION_OPTION,
        severity_benefit_option: str = DEFAULT_SEVERITY_BENEFIT_OPTION,
        radius_option: str = DEFAULT_RADIUS_OPTION,
        prop_pessimistic: float = 0,
        pessimism_level: float = 1,
        b_1: npt.NDArray[np.float64] | None = None,
        b_2: npt.NDArray[np.float64] | None = None,
        gamma_s: float = 0.001,
    ):
        """Construct new vectorised model.

        Args:
            num_agents: Number of agents in the lattice.
            width: Number of agents in the horizontal span of the lattice.
            height: Number of agents in the vertical span of the lattice.
            memory_count: Length of agents' memory.
            env_update_fn: Function which accepts the current environment and action,
                and returns an updated environment status.
            rng: Random number generator, defaults to None.
            rationality: Agent rationality. Lower --> more random. Higher --> more
                rational.
            max_storage: Maximum number of timesteps to record in history. Safeguards
                against runaway memory.
            moore: Include diagonal neighbors.
            simmer_time: Number of agent adaptation steps between environment updates.
            neighb_prediction_option: Method for predicting neighbors' actions.
            severity_benefit_option: Method for calculating the benefit of
                cooperating in a healthy environment.
            radius_option: Method for determining the radius of neighbors to consider
                when calculating neighbors' actions. Options are "single" (default) for
            prop_pessimistic: Proportion of agents to set as pessimistic.
            pessimism_level: How much pessimistic agents overestimate environmental
                degradation. Higher is more pessimistic. The default (1) is no
                pessimism.
            b_1: Initial weight for the first attribute (e.g., environmental concern).
            b_2: Initial weight for the second attribute (e.g., social norms).
            gamma_s: Rate at which agents change their action preferences.
        """
        self.time = 0
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.memory_count = memory_count
        self.env_update_fn = env_update_fn or piecewise_exponential_update(
            alpha=1, beta=1, rate=0.01
        )  # linear_update(0.05)
        self.rng = rng or np.random.default_rng()
        self.max_storage = max_storage + 1
        self.simmer_time = simmer_time
        self.neighb_prediction_option = neighb_prediction_option
        self.severity_benefit_option = severity_benefit_option
        self.radius_option = radius_option

        # Set up agents' connections and attributes
        self.adj = lattice2d(width, height, periodic=True, diagonals=moore)
        self.rationality = rationality
        self.b = np.zeros(
            (self.N_WEIGHTS, self.num_agents),
            dtype=np.float64,
        )
        if b_1 is not None and b_2 is not None:
            self.b[0] = b_1
            self.b[1] = b_2
        else:
            # Initialise weights randomly, normalised to sum to 1
            self.b[0] = self.rng.random(self.num_agents)
            self.b[1] = self.rng.random(self.num_agents)
            self.b = self.b / self.b.sum(axis=0, keepdims=True)

        self.gamma_s = gamma_s

        # Set strategy change params
        # alpha: rate of increasing support when support is low
        # beta: rate of decreasing support when support is high
        self.alpha = 1
        self.beta = 1

        pessimistic = self.rng.random(self.num_agents) < prop_pessimistic
        self.pessimism = np.ones(self.num_agents)
        self.pessimism[pessimistic] = pessimism_level

        # Initialise agents and environment
        self.initialise(zero=True)
        self.initial_action = self.action[: self.memory_count].copy()
        self.initial_environment = self.environment[: self.memory_count].copy()

    def initialise(self, zero: bool = False):
        """Initialise agents' and environment state.

        Optionally sets agents' initial actions to zero (defection), and
        environments to one (healthy).

        After initialising the environment, runs agent adaptation for 100
        steps to reach stable support levels (for cooperation) given agent
        heterogeneity.

        Args:
            zero: Initialise environment as healthy and agents as cooperating.
        """
        self.action = -np.ones(
            (self.max_storage, self.num_agents),
            dtype=np.int64,
        )
        self.environment = np.ones(
            (self.max_storage, self.num_agents),
            dtype=np.float64,
        )
        self.s = np.zeros(
            (self.max_storage, self.num_agents),
            dtype=np.float64,
        )
        self.curr_s = np.zeros(self.num_agents, dtype=np.float64) + 0.5
        if not zero:
            self.action[: self.memory_count] = self.rng.choice(
                self.ACTIONS,
                size=(self.memory_count, self.num_agents),
            )
            self.environment[: self.memory_count] = self.rng.random(
                size=(self.memory_count, self.num_agents), dtype=np.float64
            )
        for _ in range(100):
            self.adapt(self.environment[0])

    def reset(self):
        """Re-initialise agents to original actions and environment states."""
        self.initialise(zero=True)
        self.action[: self.memory_count] = self.initial_action
        self.environment[: self.memory_count] = self.initial_environment

    def run(self, steps: int = 20):
        """Run simulation for specified number of steps.

        Args:
            steps: Number of steps to iterate.
        """
        if steps > self.environment.shape[0]:
            raise RuntimeError("Maximum steps exceeded. Raise `max_storage`.")
        self.r = self.rng.random(size=(steps * self.simmer_time, self.num_agents))
        for _ in range(steps):
            self.step()

    def step(self):
        """Execute a single simulation step.

        A step comprises the following processes:
        1. Increment the simulation time.
        2. Update the environment based on agents' actions in the previous timestep.
        3. Run a fixed number of agent decision-making and adaptation steps.
        """
        self.time += 1
        self.update_env()
        self.simmer()
        self.adapt(self.environment[self.time - 1])
        self.s[self.time] = self.curr_s.copy()

    def update_env(self):
        """Update each agents' environment based on their last action."""
        self.environment[self.time] = self.env_update_fn(
            self.environment[self.time - 1],
            self.action[self.time - 1],
        )

    def simmer(self):
        """Simulate a number of agent decision-making and adaptation steps.

        A step comprises the following processes:

        1. Agents choose an action based on their current support for cooperation,
            and the social norms imposed by their neighbors.

        2. Agents adapt their support for cooperation based on the current state
            of the environment.

        Note that the environment is fixed during this process. As such, a longer
        simmer time reflects a faster rate of behavioural change relative to the
        rate of environmental change.
        """
        for i in range(self.simmer_time):
            self.decide(i)

    def decide(self, i: int):
        """Select a new action for each agent.

        The probability of selecting each action is set by an agents' logit model,
        based on their current environment and the social norms imposed by their
        neighbors.

        To select a new action, we sample a random number in [0,1] for each agent.
        If it does not exceed the probability of cooperation, the agent cooperates,
        and defects otherwise.

        Args:
            i: Simmer step idx
        """
        pa = self.action_probabilities()
        r = self.r[((self.time - 1) * self.simmer_time) + i]
        self.action[self.time] = np.where(r < pa[1], 1, -1)

    def adapt(self, n: npt.NDArray[np.float64]):
        r"""Update agents' support for cooperation.

        Agents' support for cooperation changes as a function of the current 
        environment. It decreases when the environment is either particularly
        healthy (no reason to act) or particularly unhealthy (no point in acting).
        It increases when the environment is not at either of these extremes.

        We write the change in support as a derivative:

        .. math::
            
            \frac{ds_i}{dt} = \alpha_i \sigma(n_i) (1 - s_i(t)) \
            - \beta_i (1 - \sigma(n_i)) s_i(t)

        where :math:`\sigma(n_i) = 4n_i (1 - n_i)`.

        Args:
            n: Current state of the environment, with shape (agent,)
        """
        logistic = 4 * n * (1 - n)  # Scale derivative so it is zero at the boundaries
        ds_dt = (
            self.alpha * logistic * (4 - self.curr_s)
            - self.beta * (1 - logistic) * self.curr_s
        )
        self.curr_s += self.gamma_s * ds_dt

    def pred_neighb_action(self) -> npt.NDArray[np.float64]:
        """Predict the average action of peers based on their recent actions.

        Args:
            memory (int): Number of previous steps to use for prediction.
            method (str): Prediction method, "linear" or "logistic".

        Returns:
            npt.NDArray[np.float64]: Predicted average action of
            neighbors for each agent.
        """
        # Get the recent actions for each agent (shape: memory, num_agents)
        start = max(0, self.time - self.memory_count)
        stop = self.time
        actions = self.action[start:stop]  # shape: (memory, num_agents)

        if actions.shape[0] < 2:
            return self.mean_local_action(memory=self.memory_count)

        if self.neighb_prediction_option == "linear":
            time_steps = np.arange(actions.shape[0])

            coeffs = np.polyfit(time_steps, actions, 1)

            predicted = np.polyval(coeffs, actions.shape[0])

            neighb_pred = self.adj @ predicted

            return np.where(neighb_pred >= 0, 1, -1)
        elif self.neighb_prediction_option == "logistic":
            # For logistic regression, map actions from -1/1 to 0/1
            act_bin = (actions + 1) // 2
            time_steps = np.arange(actions.shape[0])
            log_time = np.log(time_steps + 1)
            predicted_probs = np.zeros(self.num_agents)
            for i in range(self.num_agents):
                try:
                    popt, _ = scipy.optimize.curve_fit(
                        lambda t, a, b, c: a / (1 + np.exp(-b * (t - c))),
                        log_time,
                        act_bin[:, i],
                        maxfev=10000,
                        bounds=([0, -np.inf, -np.inf], [1, np.inf, np.inf]),
                    )
                    pred_prob = popt[0] / (
                        1 + np.exp(-popt[1] * (np.log(actions.shape[0] + 1) - popt[2]))
                    )
                except Exception:
                    pred_prob = np.mean(act_bin[:, i])
                predicted_probs[i] = pred_prob
            neighb_pred = self.adj @ predicted_probs
            return np.where(neighb_pred >= 0.5, 1, -1)
        else:
            return self.mean_local_action(memory=self.memory_count)

    def action_probabilities(self) -> npt.NDArray[np.float64]:
        r"""Calculate the probability of each possible action.

        The probabilities are calculated using a logit softmax
        function over the utilities of each action.
        The formula is:

        .. math::

            P(a_i(t) = a) = \frac{\exp(\lambda \cdot V_i(a))}{\exp(\lambda \cdot \
            V_i(C)) + \exp(\lambda \cdot V_i(D))}

        Where :math:`V_i(a)` is the representative utility for action :math:`a`
        for agent :math:`i`.

        Returns:
            An array of probabilities for each action, with shape (2,agent).
        """
        if self.severity_benefit_option == "adaptive":
            m = self.pred_neighb_action()
            z = self.b[0] * (self.curr_s - 2) + 2 * self.b[1] * m
            pc = 1 / (1 + np.exp(-2 * self.rationality * z))
            return (1 - pc, pc)

        utilities = self.rationality * np.array(
            [self.representative_utility(-1), self.representative_utility(1)]
        )
        exp_utilities = np.exp(utilities - np.max(utilities, axis=0))
        return exp_utilities / exp_utilities.sum(axis=0)

    def representative_utility(self, action: int) -> float:
        r"""Calculate the representative utility of an action for each agent.

        Representative utility is a linear combination of the support for 
        cooperation and the social norms imposed by an agents' neighbors:

        .. math::
            
            V_i(a) = b_1 \cdot [a^* \cdot s_i(t) + (1 - a^*) \cdot (1 - s_i(t))] + \
            b_2 (a^* - \overline{A^*}_i (t))^2

        Where :math:`a^* = (a + 1)/2` is a transformation of the action to the set 
        :math:`\{0,1\}`.
        """
        if self.severity_benefit_option == "adaptive":
            a = int((action + 1) / 2)
            severity_benefit = (a) * self.curr_s + (1 - a) * (4 - self.curr_s)
        else:
            severity_benefit = -(2 * self.environment[self.time - 1] - 1) * action

        deviation_cost = (action - self.pred_neighb_action()) ** 2

        return self.b[0] * severity_benefit - self.b[1] * deviation_cost

    def mean_local_action(self, memory: int = 1) -> npt.NDArray[np.float64]:
        """Calculate the average action in each agents' local neighborhood.

        Args:
            memory: Number of previous neighbors' actions to consider.

        Returns:
            The mean local action for each agent, reflecting the perceived social norm
            for each agent at the current timestep. Shape is (agent,).
        """
        if self.radius_option == "single":
            if memory == -1:
                memory = self.max_storage
            start = max(0, self.time - memory)
            stop = self.time
            mean_per_timestep = self.adj @ self.action[start:stop].T
            return mean_per_timestep.mean(axis=1)
        elif self.radius_option == "all":
            if memory == -1:
                memory = self.max_storage
            start = max(0, self.time - memory)
            stop = self.time
            mean_per_timestep = self.action[start:stop].mean(axis=1)
            return np.full(self.num_agents, mean_per_timestep.mean())

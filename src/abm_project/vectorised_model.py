"""Vectorised implementation of the OOP model."""

import itertools
from collections.abc import Callable

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy

from .oop_model import BaseModel

type EnvUpdateFn = Callable[(npt.NDArray[float], npt.NDArray[int]), npt.NDArray[float]]


def sigmoid_update(n: npt.NDArray[float], a: npt.NDArray[int]) -> npt.NDArray[float]:
    """Update environment according to sigmoid rule.

    Args:
        n: Each agents' current environment, shape: (agents,)
        a: Each agents' current action, shape: (agents,)

    Returns:
        Numpy array of new environment values for each agent, with same shape as `n`.
    """
    sensitivity = 1 / (1 + np.exp(6 * (n - 0.5)))
    delta = sensitivity * a * 0.05
    return n + delta


def exponential_update(rate: float):
    r"""Construct exponential environment update function.

    .. math::

        n(t+1) = n(t) + a\cdot r \cdot \exp(-n(t))

    Args:
        rate: Multiplicative coefficient for exponential function.

    Returns:
        Exponential update function.
    """

    def inner(n: npt.NDArray[float], a: npt.NDArray[int]) -> npt.NDArray[float]:
        """Update environment according to exponential rule.

        Args:
            n: Each agents' current environment, shape: (agents,)
            a: Each agents' current action, shape: (agents,)

        Returns:
            Numpy array of new environment values for each agent, with same shape
            as `n`.
        """
        delta = a * rate * np.exp(-n)
        return n + delta

    return inner


def linear_update(rate: float):
    r"""Construct linear environment update function.

    .. math::

        n(t+1) = n(t) + a\cdot r

    The returned value is clipped to the interval [0,1].

    Args:
        rate: Linear step size.

    Returns:
        Linear update function.
    """

    def inner(n: npt.NDArray[float], a: npt.NDArray[int]) -> npt.NDArray[float]:
        """Update environment according to linear rule.

        Args:
            n: Each agents' current environment, shape: (agents,)
            a: Each agents' current action, shape: (agents,)

        Returns:
            Numpy array of new environment values for each agent, with same shape
            as `n`.
        """
        delta = a * rate
        return np.clip(n + delta, a_min=0.0, a_max=1.0)

    return inner


def lattice2d(width: int, height: int, periodic: bool = True, diagonals: bool = False):
    """Construct normalised adjacency matrix for a 2D lattice.

    Uses networkx to create a 2D lattice with optional periodic boundaries and
    Moore neighborhoods (diagonals). Converts this to a sparse adjacency matrix
    with normalised rows, to simplify computing averages over neighborhoods.

    Args:
        width: Number of nodes along the horizontal span of the lattice.
        height: Number of nodes along the vertical span of the lattice.
        periodic: Connect nodes at the edges of the lattice with periodic boundary
            conditions.
        diagonals: Connect nodes to their diagonal neighbors, also known as the Moore
            neighborhood. Default is the von Neumann neighborhood (cartesian neighbors).

    Returns:
        Sparse CSR adjacency matrix with shape (width x height, width x height),
        normalised per row.
    """
    network = nx.grid_2d_graph(width, height, periodic=periodic)

    if diagonals:
        for y, x in itertools.product(range(height), range(width)):
            for dy, dx in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ty, tx = y + dy, x + dx
                if periodic:
                    ty = ty % height
                    tx = tx % width
                if 0 <= ty < height and 0 <= tx < width:
                    network.add_edge((y, x), (ty, tx))

    # Extract normalised adjacency matrix
    adj = nx.adjacency_matrix(network)
    adj = adj / adj.sum(axis=1)[:, None]
    return scipy.sparse.csr_array(adj)


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
    DEFAULT_SIMMER_TIME = 0

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
        """
        self.time = 0
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.memory_count = memory_count
        self.env_update_fn = env_update_fn or linear_update(0.05)
        self.rng = rng or np.random.default_rng()
        self.max_storage = max_storage + 1 + self.memory_count
        self.simmer_time = simmer_time

        # Set up agents' connections and attributes
        self.adj = lattice2d(width, height, periodic=True, diagonals=moore)
        self.rationality = rationality
        self.b = self.rng.random((self.N_WEIGHTS, self.num_agents))
        self.b = np.full_like(self.b, fill_value=1)
        self.b = self.b / self.b.sum(axis=1, keepdims=True)  # Normalise

        # Initialise agents and environment
        self.initialise()
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
        self.action = np.zeros(
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
        for _ in range(steps):
            self.step()

    def step(self):
        """Execute a single simulation step.

        A step comprises the following processes:
        1. Increment the simulation time.
        2. Update the environment based on agents' actions in the previous timestep.
        3. Run a fixed number of agent decision-making and adaptation steps.
        """
        if self.time > self.environment.shape[0]:
            raise RuntimeError("Maximum steps exceeded. Raise `max_storage`.")

        self.time += 1
        self.update_env()
        self.simmer()
        self.s[self.time] = self.curr_s.copy()
        # self.decide()

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
        for _ in range(self.simmer_time):
            self.decide()
            self.adapt(self.environment[self.time - 1])

    def decide(self):
        """Select a new action for each agent.

        The probability of selecting each action is set by an agents' logit model,
        based on their current environment and the social norms imposed by their
        neighbors.

        To select a new action, we sample a random number in [0,1] for each agent.
        If it does not exceed the probability of cooperation, the agent cooperates,
        and defects otherwise.
        """
        pa = self.action_probabilities()
        r = self.rng.random(size=self.num_agents)
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
        ds_dt = 1 * logistic * (1 - self.curr_s) - 1 * (1 - logistic) * self.curr_s
        self.curr_s += 0.01 * ds_dt

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
        # severity_benefit = -(2 * self.environment[self.time - 1] - 1) * action
        a = (action + 1) / 2
        severity_benefit = (a) * self.curr_s + (1 - a) * (1 - self.curr_s)
        deviation_cost = ((action - self.mean_local_action()) / 2) ** 2
        return self.b[0] * severity_benefit - self.b[1] * deviation_cost

    def mean_local_action(self, memory: int = 1) -> npt.NDArray[np.float64]:
        """Calculate the average action in each agents' local neighborhood.

        Args:
            memory: Number of previous neighbors' actions to consider.

        Returns:
            The mean local action for each agent, reflecting the perceived social norm
            for each agent at the current timestep. Shape is (agent,).
        """
        if memory == -1:
            memory = self.max_storage
        start = max(0, self.time - memory)
        stop = self.time
        mean_per_timestep = self.adj @ self.action[start:stop].T
        return mean_per_timestep.mean(axis=1)

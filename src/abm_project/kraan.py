"""Implementation of Kraan 2D lattice model."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


class KraanModel:
    """Agent-based decision model for energy transition.

    Agents decide to 'act' or 'not act' at each timestep based on the state of the
    environment and the social pressures of their neighbors.
    """

    def __init__(self, width: int, height: int, c: float, seed: int, n_update_fn):
        """Initialise model."""
        # System attributes
        self.rng = np.random.default_rng(seed)
        self.n_agents = width * height

        # Agent attributes
        self.c = c
        self.u_inactive = 2

        # Environment attributes and update fn
        self.n = -1
        self.update_env = n_update_fn

        # Create network and normalised-influence adjacency matrix
        self.network = nx.grid_2d_graph(width, height, periodic=True)
        adj = nx.adjacency_matrix(self.network)
        self.adj = adj / adj.sum(axis=1)[:, None]

        # Initialise agents' actions to -1
        self.action = np.full(self.n_agents, fill_value=-1)

    def run(self, update_steps: int, simmer_steps: int):
        """Run model for a number of environmental updates.

        Runs the model for a number of steps. Each step comprises two parts:
        1. Update the environment,
        2. Simulate a number of decision steps, so as to reach equilibrium.

        Finally, we yield control to the caller, passing the current timestep.

        Args:
            update_steps: Number of times to update the environment.
            simmer_steps: Number of decision steps to simulate after each environment
                update.

        Yields:
            The current timestep, measured in number of environmental updates (minus 1).
        """
        for t in range(update_steps):
            self.update_env(self, t)
            self.simmer(simmer_steps)
            yield t

    def simmer(self, steps):
        """Simulate a number of decisions for each agent.

        This is a convenience method for allowing the model to reach equilibrium
        after an environment update.

        Args:
            steps: Number of decisions to simulate.
        """
        for _ in range(steps):
            self.decide()

    def decide(self):
        """Simulate a single decision step for each agent.

        Agents' decisions are probabilistically sampled according to a logistic
        model, with representative utility of an action $V_i(a)$ balancing an agents'
        current perception of the environment, and their social pressures.
        """
        mean_neighbor_action = self.adj @ self.action

        # Calculate each agent's probability of being active
        v_active = self.u_active - self.c * (1 - mean_neighbor_action) ** 2
        v_inactive = self.u_inactive - self.c * (1 + mean_neighbor_action) ** 2
        z = np.exp(v_active) + np.exp(v_inactive)
        probability_active = np.exp(v_active) / z

        # Sample new actions
        self.action = self.choose(probability_active)

    @property
    def u_active(self) -> float:
        r"""Calculate the utility for an agent to be active in the current state.

        The environment is defined by

        $h = \frac{U(\text{active}) + U(\text{inactive})}{2}$

        Since the utility for inaction is fixed, we calculate the utility
        for action by rearranging this formula.

        Returns:
            The current (homogeneous) utility for action.
        """
        return 2 * self.n + self.u_inactive

    def choose(self, p_active) -> npt.NDArray[np.int64]:
        r"""Sample a decision for each agent.

        After sampling a random number $r \in [0,1]$, decisions are taken to
        be +1 (action) if $r < p\_\text{active}$ for a given agent, and -1 (inactive)
        otherwise.

        Args:
            p_active: 1D numpy array with length equal to the number of agents,
                containing the probability for each agent choosing to act.

        Returns:
            A 1D numpy array containing the sampled action for each agent.
        """
        r = self.rng.random(size=len(p_active))
        return np.where(r < p_active, 1, -1)

    def draw(self):
        """Plot the current state of the network and agents' actions."""
        color_state_map = {-1: "red", 1: "green"}
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.network, seed=42)
        node_colors = [color_state_map[action] for action in self.action]

        nx.draw(
            self.network, pos, node_size=50, node_color=node_colors, with_labels=False
        )
        plt.show()


def exogenous_env(
    n0: float = -1,
    n_max: float = 1,
    increment_steps: int = 40,
    decrement_steps: int = 40,
):
    """Construct an exogenous environment update function.

    Comprises a sequence of linear increments to the environment, followed by a
    sequence of linear decrements, returning to the original value.

    Default arguments are as specified in Kraan 2019.

    Args:
        n0: Initial state of the environment at time $t=0$.
        n_max: Maximum value of the environment, to be reached after increments.
        increment_steps: Number of steps to take when increasing environment state to
            n_max.
        decrement_steps: Number of steps to take when decrementing environment state
            from n_max to n0.

    Returns:
        A constructed function to execute the specified update strategy.
    """
    h_inc = (n_max - n0) / increment_steps
    h_dec = (n_max - n0) / decrement_steps

    def inner(model: KraanModel, t: int):
        if t == 0:
            model.n = n0
        elif t <= increment_steps:
            model.n += h_inc
        elif increment_steps < t <= increment_steps + decrement_steps:
            model.n -= h_dec
        else:
            model.n = n0

    return inner

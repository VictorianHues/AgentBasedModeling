"""Utility functions for agent-based models."""

import itertools
from collections.abc import Callable

import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy

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


def piecewise_exponential_update(recovery: float, pollution: float, gamma: float):
    r"""Construct piecewise exponential environment update function.

    Args:
        recovery: Rate of improvement due to good actions
        pollution: Rate of degradation due to bad actions
        gamma: Step size for environmental change

    Returns:
        Piecewise exponential update function.
    """

    def inner(n: npt.NDArray[float], a: npt.NDArray[int]) -> npt.NDArray[float]:
        """Update environment according to piecewise exponential rule.

        Args:
            n: Each agents' current environment, shape: (agents,)
            a: Each agents' current action, shape: (agents,)

        Returns:
            Numpy array of new environment values for each agent, with same shape
            as `n`.
        """
        pc = (a + 1) / 2
        improvement = recovery * (1 - n) * pc
        degradation = pollution * n * (1 - pc)
        dn_dt = gamma * (improvement - degradation)
        return n + dn_dt

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

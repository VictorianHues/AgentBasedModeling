"""Mean-field model calculations and simulation."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


def compute_s_from_p(p: float, b: float, c: float) -> float:
    """Calculate mean preference for cooperation given mean P(C)."""
    return (1 / b) * (2 * (b + c) - 4 * c * p - (1 / 2) * np.log((1 - p) / p))


# def compute_s_from_m(m: float, rationality: float, b: float, c: float) -> float:
#    return (1 / (rationality * b)) * np.arctanh(m) + 2 * (1 - (c / b) * m)


def f_dn_dt(recovery: float, pollution: float, rate: float = 0.01):
    """Construct parameterised mean-field dn/dt derivative function.

    Calculates rate of change in the mean environmental state, given the
    expected probability of cooperation in a mean-field model.

    Args:
        recovery: How quickly the environment recovers under positive action.
        pollution: How quickly the environment degrades due to negative action.
        rate: Scale coefficient for the derivative, controls the general rate
            of change in the environment.
    """

    def derivative(n: float, p: float) -> float:
        return rate * (recovery * (1 - n) * p - pollution * n * (1 - p))

    return derivative


def f_ds_dt(alpha: float, beta: float, rate: float = 0.001):
    """Construct parameterised mean-field ds/dt derivative function.

    Calculates rate of change in the average preference for cooperation, given
    the current average state of the average environment in a mean-field model.

    Args:
        alpha: How quickly agents increase support for climate mitigation when the
            environment is non-extreme.
        beta: How quickly agents decrease support for climate mitigation when the
            environment is particularly good (no reason to act) or particularly
            bad (action is meaningless).
        rate: Scale coefficient for the derivative, controls the general rate
            of change in preference for cooperation.
    """

    def sigma(n: float) -> float:
        return 4 * n * (1 - n)

    def derivative(n: float, s: float) -> float:
        return rate * (alpha * sigma(n) * (4 - s) - beta * (1 - sigma(n)) * s)

    return derivative


# def f_dp_dt(b: float, c: float, alpha: float, beta: float, rate: float = 0.001):
#    s_prime = f_ds_dt(alpha, beta, rate)
#
#    def derivative(p: float, n: float) -> float:
#        s = compute_s_from_p(p, b, c)
#        ds_dt = s_prime(n, s)
#        dp_ds = (2 * b * p * (1 - p)) / (1 - 8 * c * p * (1 - p))
#        dp_dt = dp_ds * ds_dt
#        return dp_dt
#
#    return derivative


def f_dm_dt(
    rationality: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    rate: float = 0.001,
):
    """Construct parameterised mean-field dm/dt derivative function.

    Calculates rate of change in the average action, given the current average state
    of the environment, and average action in a mean-field model.

    Args:
        rationality: Controls how rational agents are. Larger is more rational
            (deterministic). 0 is random.
        b: Utility function weight for the 'individual action preference' term.
        c: Utility function weight for the 'peer pressure' term.
        alpha: How quickly agents increase support for climate mitigation when the
            environment is non-extreme.
        beta: How quickly agents decrease support for climate mitigation when the
            environment is particularly good (no reason to act) or particularly
            bad (action is meaningless).
        rate: Scale coefficient for the derivative, controls the general rate
            of change in preference for cooperation.
    """
    s_prime = f_ds_dt(alpha, beta, rate)

    def calculate_s(m: float) -> float:
        return (1 / (rationality * b)) * np.arctanh(m) + 2 * (1 - (c / b) * m)

    def calculate_z(m: float, s: float) -> float:
        return b * (s - 2) + 2 * c * m

    def derivative(m: float, n: float) -> float:
        s = calculate_s(m)
        z = calculate_z(m, s)
        ds_dt = s_prime(n, s)
        sech_term = 1 / np.cosh(rationality * z) ** 2
        numerator = b * rationality * sech_term * ds_dt
        denominator = 1 - 2 * rationality * c * sech_term
        return numerator / denominator

    return derivative


def solve_for_equilibria(
    b: float,
    c: float,
    rationality: float,
    recovery: float,
    pollution: float,
    alpha: float = 1,
    beta: float = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Identify mean-field equilibrium points for a model.

    Equilibria are characterised by the state of the environment, and the mean action.
    A point (n,m) is an equilibria if dn/dt = dm/dt = 0.

    Args:
        b: Utility function weight for the 'individual action preference' term.
        c: Utility function weight for the 'peer pressure' term.
        rationality: Controls how rational agents are. Larger is more rational
            (deterministic). 0 is random.
        alpha: How quickly agents increase support for climate mitigation when the
            environment is non-extreme.
        beta: How quickly agents decrease support for climate mitigation when the
            environment is particularly good (no reason to act) or particularly
            bad (action is meaningless).
        recovery: How quickly the environment recovers under positive action.
        pollution: How quickly the environment degrades due to negative action.
        rate: Scale coefficient for the derivative, controls the general rate
            of change in preference for cooperation.

    Returns:
        A tuple (N, M) containing pairs of equilibria points
        (environment state, action).
    """

    # Determine P(C) s.t. dP/dt = 0
    def calc_pc_stationary_n(n: float):
        return n / (recovery + n * (pollution - recovery))

    # Determine S(P*(C)) s.t. dP/dt = 0
    def calc_s_stationary_n(n: float):
        pc = calc_pc_stationary_n(n)
        log_odds = np.log(1 - pc) - np.log(pc)
        return (1 / b) * (2 * (b + c) - 4 * c * pc - (1 / (2 * rationality)) * log_odds)

    def calc_s_stationary_s(n: float) -> float:
        sigma_n = 4 * n * (1 - n)
        s = 4 * alpha * sigma_n / (beta * (1 - sigma_n) + alpha * sigma_n)
        return s

    def f(n: float) -> float:
        return calc_s_stationary_s(n) - calc_s_stationary_n(n)

    # Try to find lower fixed point
    n_lower = n_middle = n_upper = None
    fp_lower = root_scalar(f, x0=1e-5)
    if not fp_lower.converged:
        print("Warning: No lower fixed-point found.")
    elif 0.0 < fp_lower.root < 1.0:
        if fp_lower.root > 0.5:
            n_upper = float(fp_lower.root)
        else:
            n_lower = float(fp_lower.root)

    fp_upper = root_scalar(f, x0=1 - 1e-5)
    if not fp_upper.converged:
        print("Warning: No upper fixed-point found.")
    elif 0.0 < fp_upper.root < 1.0:
        if fp_upper.root < 0.5:
            n_lower = float(fp_upper.root)
        else:
            n_upper = float(fp_upper.root)

    if n_lower and n_upper:
        # Look for central root
        fp_middle = root_scalar(
            f, method="bisect", bracket=(n_lower + 1e-3, n_upper - 1e-3)
        )
        if not fp_middle.converged:
            print("Warning: Bisection method failed to converge for middle root.")
        elif not (n_lower < fp_middle.root <= n_upper):
            print(
                "Warning: Middle root found outside expected range: "
                f"{fp_middle.root} not in ({n_lower:.2f}, {n_upper:.2f})."
            )
        else:
            n_middle = float(fp_middle.root)

    ns = []
    for n in (n_lower, n_middle, n_upper):
        if n:
            ns.append(n)
    ns = np.asarray(ns)
    pc = ns / (recovery + ns * (pollution - recovery))
    ms = 2 * pc - 1
    return ns, ms


@dataclass
class FixedpointResult:
    """Solutions to a mean-action fixed point problem."""

    lower: float | None = None
    middle: float | None = None
    upper: float | None = None

    def __len__(self):
        """Number of fixed-points identified."""
        return sum(map(lambda x: x is not None, (self.lower, self.middle, self.upper)))

    def stable(self) -> list[float]:
        """Retrieve the stable fixed points.

        Returns:
            A list of floats representing each stable fixed point.
        """
        roots = []
        if self.lower:
            roots.append(self.lower)
        if self.upper:
            roots.append(self.upper)
        return roots

    def unstable(self) -> list[float]:
        """Retrieve the unstable fixed points. May be empty.

        Returns:
            A (possibly empty) list of floats representing each unstable fixed point.
        """
        return [self.middle] if self.middle else []

    def roots(self) -> list[float]:
        """Retrieve the fixed points.

        Returns:
            A list of floats representing each fixed point, both stable and unstable,
            in increasing value order.
        """
        roots = []
        if self.lower:
            roots.append(self.lower)
        if self.middle:
            roots.append(self.middle)
        if self.upper:
            roots.append(self.upper)
        return roots


def fixedpoint_mean_action(
    s: float,
    c: float,
    rationality: float = 1,
    ignore_warnings: bool = False,
) -> FixedpointResult:
    """Find all possible mean-action values, given mean preference for cooperation.

    Args:
        s: Average preference for climate mitigation in the mean-field model.
        c: Utility function weight for the 'peer pressure' term.
        rationality: Controls how rational agents are. Larger is more rational
            (deterministic). 0 is random.
        ignore_warnings: Don't print warning messages when fixed-point solver doesn't
            converge.

    Returns:
        A FixedPointResult object containing all possible values which the mean action
        can take.
    """

    def g(m: float) -> float:
        return np.tanh(rationality * (1 - c) * (s - 2) + 2 * rationality * c * m) - m

    def gprime(m: float) -> float:
        t = np.tanh(rationality * (1 - c) * (s - 2) + 2 * rationality * c * m)
        return 2 * rationality * c * (1 - t**2) - 1

    def multiple_fixed_points_possible() -> bool:
        return 1 / (2 * rationality * c) < 1

    def find_turning_points():
        """Fixed points have form m0 +- B."""
        m0 = -(1 / (2 * c)) * (1 - c) * (s - 2)
        b = (1 / (2 * rationality * c)) * np.arccosh(np.sqrt(2 * rationality * c))
        return (m0 - b, m0 + b)

    # If multiple fixed points not possible, find unique FP
    if not multiple_fixed_points_possible():
        root_res = root_scalar(g, x0=0.0, fprime=gprime)
        if not root_res.converged:
            if not ignore_warnings:
                print(
                    f"Warning: Root-finding didn't converge: {s=}, {c=}, {rationality=}"
                )
        elif not (-1 <= root_res.root <= 1):
            print(
                f"Warning: Root found outside expected range "
                f"(m={root_res.root}): {s=}, {c=}, {rationality=}"
            )
        root = root_res.root
        if root < 0:
            return FixedpointResult(lower=root)
        else:
            return FixedpointResult(upper=root)

    # Determine existence, stability of upper and lower fixed points
    # 1. Check that turning point exists within [-1, 1]
    # 2. Check that turning point corresponds to intersection with y=x, i.e.,
    #       lower FP < 0, upper FP > 0
    # Note: g(tp) == 0 implies fixed point at tp, however, this FP is unstable
    #       so we ignore it here.
    fp_lower_exists = fp_upper_exists = False
    tp_lower, tp_upper = find_turning_points()
    if tp_lower >= -1 and g(tp_lower) < 0:
        fp_lower_exists = True
    if tp_upper <= 1 and g(tp_upper) > 0:
        fp_upper_exists = True

    # Solve for the fixed points
    fp_lower = fp_upper = None
    # Lower
    if fp_lower_exists:
        lower_root_res = root_scalar(g, x0=-1.0, fprime=gprime)
        if not lower_root_res.converged:
            if not ignore_warnings:
                print(
                    f"Warning: Lower root-finding didn't converge: "
                    f"{s=}, {c=}, {rationality=}"
                )
        elif not (-1 <= lower_root_res.root <= 0):
            print(
                f"Warning: Lower root found outside expected range "
                f"(m={lower_root_res.root}): {s=}, {c=}, {rationality=}"
            )
        else:
            fp_lower = lower_root_res.root

    # Upper
    if fp_upper_exists:
        upper_root_res = root_scalar(g, x0=1.0, fprime=gprime)
        if not upper_root_res.converged:
            if not ignore_warnings:
                print(
                    f"Warning: Upper root-finding didn't converge: "
                    f"{s=}, {c=}, {rationality=}"
                )
        elif not (0 <= upper_root_res.root <= 1):
            print(
                f"Warning: Upper root found outside expected range "
                f"(m={upper_root_res.root}): {s=}, {c=}, {rationality=}"
            )
        else:
            fp_upper = upper_root_res.root

    # If we have two fixed points, there must be an unstable third between them
    fp_middle = None
    if fp_lower and fp_upper:
        middle_root_res = root_scalar(
            g, method="bisect", bracket=(fp_lower + 1e-3, fp_upper - 1e-3)
        )
        if not middle_root_res.converged:
            if not ignore_warnings:
                print(
                    f"Warning: Bisection method failed to converge. "
                    f"Expected unstable middle root in ({fp_lower:.2f},"
                    f"{fp_upper:.2f}) for parameters {s=}, {c=}, {rationality=}."
                )
        elif not (fp_lower < middle_root_res.root <= fp_upper):
            print(
                "Warning: Unstable middle root found outside expected range: "
                f"{middle_root_res.root} not in ({fp_lower:.2f}, {fp_upper:.2f})."
            )
        else:
            fp_middle = middle_root_res.root

    return FixedpointResult(lower=fp_lower, middle=fp_middle, upper=fp_upper)


# def solve_for_fixedpoint_abar(s: float, c: float, ignore_warnings: bool = False):
#    r"""Identify fixed-point values of :math:`\bar{a}` for given :math:`S_c`."""
#
#    def f(abar):
#        return np.tanh((1 - c) * (s - 2) + 2 * c * abar) - abar
#
#    def fprime(abar):
#        t = np.tanh((1 - c) * (s - 2) + 2 * c * abar)
#        return 2 * c * (1 - t**2) - 1
#
#    roots = []
#    lower_root_res = root_scalar(f, x0=-1.0, fprime=fprime)
#    if not lower_root_res.converged and not ignore_warnings:
#        print(f"Warning: Lower root-finding didn't converge: {s=}, {c=}")
#    if 0 <= lower_root_res.root <= 1.0:
#        return [lower_root_res.root]
#    elif -1.0 <= lower_root_res.root <= 1.0:
#        # Make sure root is stable. Below: -ve, above: +ve
#        below = f(lower_root_res.root - 0.01)
#        above = f(lower_root_res.root + 0.01)
#        if below > 0 and above < 0:
#            roots.append(lower_root_res.root)
#
#    upper_root_res = root_scalar(f, x0=1.0, fprime=fprime)
#    if not upper_root_res.converged and not ignore_warnings:
#        print(f"Warning: Upper root-finding didn't converge: {s=}, {c=}")
#    if -1.0 <= upper_root_res.root <= 0.0:
#        # No upper root, return solution for lower root
#        return roots
#    elif -1.0 <= upper_root_res.root <= 1.0:
#        # Make sure its stable
#        below = f(upper_root_res.root - 0.01)
#        above = f(upper_root_res.root + 0.01)
#        if below > 0 and above < 0:
#            roots.append(upper_root_res.root)
#    return roots


# def solve_for_fixedpoint_pc(
#    s: float, b0: float, b1: float, ignore_warnings: bool = False
# ):
#    """Identify fixed-point values of :math:`P_c` for given :math:`S_c`."""
#
#    def fexp(pc: float):
#        return np.exp(-2 * (b0 * s + 4 * b1 * pc - 2 * b1 - 2 * b0))
#
#    def f(pc):
#        return 1 / (1 + fexp(pc)) - pc
#
#    def fprime(pc):
#        exp_part = fexp(pc)
#        return (8 * b1 * exp_part) / (1 + exp_part) ** 2 - 1
#
#    roots = []
#    lower_root_res = root_scalar(f, x0=0.0, fprime=fprime)
#    if not lower_root_res.converged and not ignore_warnings:
#        print(f"Warning: Lower root-finding didn't converge: {s=}, {b0=}, {b1=}")
#    if 0.5 <= lower_root_res.root <= 1.0:
#        return [lower_root_res.root]
#    elif 0.0 <= lower_root_res.root <= 1.0:
#        # Make sure root is stable. Below: -ve, above: +ve
#        below = f(lower_root_res.root - 0.01)
#        above = f(lower_root_res.root + 0.01)
#        if below > 0 and above < 0:
#            roots.append(lower_root_res.root)
#
#    upper_root_res = root_scalar(f, x0=1.0, fprime=fprime)
#    if not upper_root_res.converged and not ignore_warnings:
#        print(f"Warning: Upper root-finding didn't converge: {s=}, {b0=}, {b1=}")
#    if 0.0 <= upper_root_res.root <= 0.5:
#        # No upper root, return solution for lower root
#        return roots
#    elif 0.0 <= upper_root_res.root <= 1.0:
#        # Make sure its stable
#        below = f(upper_root_res.root - 0.01)
#        above = f(upper_root_res.root + 0.01)
#        if below > 0 and above < 0:
#            roots.append(upper_root_res.root)
#    return roots


def solve(
    b: float,
    c: float,
    alpha: float,
    beta: float,
    pollution: float,
    recovery: float,
    n_update_rate: float,
    s_update_rate: float,
    n0: float | int,
    m0: float | int,
    num_steps: int,
    rationality: float = 1.0,
):
    """Simulate a mean-field model run.

    Args:
        b: Utility function weight for the 'individual action preference' term.
        c: Utility function weight for the 'peer pressure' term.
        alpha: How quickly agents increase support for climate mitigation when the
            environment is non-extreme.
        beta: How quickly agents decrease support for climate mitigation when the
            environment is particularly good (no reason to act) or particularly
            bad (action is meaningless).
        pollution: How quickly the environment degrades due to negative action.
        recovery: How quickly the environment recovers under positive action.
        n_update_rate: Scale coefficient for dn/dt, controls the general rate
            of change in the (average) environment.
        s_update_rate: Scale coefficient for ds/dt, controls the general rate
            of change in preference for cooperation.
        n0: Initial average environmental state.
        m0: Initial average action.
        num_steps: Number of time-steps to simulate.
        rationality: Controls how rational agents are. Larger is more rational
            (deterministic). 0 is random.

    Returns:
        A tuple (t, results), where t is a vector of real-values time points at
        which the model state is measured, and results is a tuple (n, s, sp, m, p),
        where: n is the mean environment state, s is the mean preference for climate
        mitigation, sp is the mean social pressure experienced at each timestep, m
        is the mean action, and p is the probability of choosing climate mitigation.
    """
    dn_dt = f_dn_dt(recovery, pollution, n_update_rate)
    dm_dt = f_dm_dt(rationality, b, c, alpha, beta, s_update_rate)

    def f(t, y):
        n, m = y
        return (dn_dt(n, (m + 1) / 2), dm_dt(m, n))

    t_eval = np.arange(num_steps + 1)
    sol = solve_ivp(f, t_span=(0, num_steps), y0=(n0, m0), t_eval=t_eval)

    n = sol.y[0]  # Environment
    m = sol.y[1]  # Probability of cooperating
    p = (m + 1) / 2
    sp = p * c * (1 - p) ** 2 + (1 - p) * c * (1 + p) ** 2  # Social pressure
    s = compute_s_from_p(p, b, c)  # Willingness to cooperate

    return sol.t, (n, s, sp, m, p)

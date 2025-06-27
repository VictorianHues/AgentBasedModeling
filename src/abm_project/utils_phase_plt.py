"""This module contains functions to compute and plot the phase portrait."""

# utils_plotting.py
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from abm_project.utils import piecewise_exponential_update  # pick the one you need
from abm_project.vectorised_model import VectorisedModel


# --------------------------------------------------------------------------- #
# 1.  ── helper that gives Δn, Δm at a single point (n0,m0)  ──────────────── #
# --------------------------------------------------------------------------- #
def flow_at_point(n0, m0, *, params, n_agents=2_500, reps=3):
    """Estimate one-step changes (Δn, Δm) at the macroscopic state (n0, m0).

    It:
    • builds a *fresh* model with the same micro-parameters for every call
    • initialises the agents so that           mean(environment)=n0,  mean(action)=m0
    • advances the simulation by exactly one *VectorisedModel.step()*
    • averages the deltas over `reps` independent restarts

    Args:
        n0: Initial environment value (mean).
        m0: Initial mean action value.
        params: Dictionary of micro-parameters for the ABM.
        n_agents: Number of agents in the model.
        reps: Number of independent restarts to average over.

    Returns:
        Δn_acc: Average change in environment after one step.
        Δm_acc: Average change in mean action after one step.
    """
    Δn_acc = 0.0
    Δm_acc = 0.0

    # probability that a random agent is a co-operator, given mean action m0
    p_coop = (m0 + 1) / 2

    for _ in range(reps):
        mdl = VectorisedModel(
            width=params["width"],
            height=params["height"],
            num_agents=n_agents,
            memory_count=params["memory_count"],
            rationality=params["rationality"],
            gamma_s=params["gamma_s"],
            env_update_fn=params["env_update_fn"],
            max_storage=2,  # we only need t = 0,1
        )
        # overwrite *initial* micro-state so it has the right macroscopics
        mdl.action[:] = -1
        coop_mask = mdl.rng.random(n_agents) < p_coop
        mdl.action[0, coop_mask] = +1

        mdl.environment[:] = n0

        # one full ABM step
        mdl.step()

        Δn_acc += mdl.environment[1].mean() - n0
        Δm_acc += mdl.action[1].mean() - m0

    return Δn_acc / reps, Δm_acc / reps


# --------------------------------------------------------------------------- #
# 2.  ── wrapper that computes the field on a full grid  ───────────────────── #
# --------------------------------------------------------------------------- #
def compute_field(params, grid_N=26, reps=3):
    """Compute the flow field on a grid of size grid_N x grid_N.

    Args:
        params: Dictionary of micro-parameters for the ABM.
        grid_N: Number of nodes along each axis of the grid.
        reps: Number of independent restarts for each grid node.

    Returns:
        N, M: Meshgrid of environment and mean action values.
        U, V: Flow vectors (Δn, Δm) at each grid node.
    """
    n_lin = np.linspace(0.00, 1.00, grid_N)
    m_lin = np.linspace(-1.0, 1.0, grid_N)
    N, M = np.meshgrid(n_lin, m_lin)

    U = np.zeros_like(N)  # Δn
    V = np.zeros_like(M)  # Δm

    for i, j in itertools.product(range(grid_N), range(grid_N)):
        U[j, i], V[j, i] = flow_at_point(
            N[j, i],
            M[j, i],
            params=params,
            reps=reps,
        )
    return N, M, U, V


# --------------------------------------------------------------------------- #
# 3.  ── find fixed points by root-finding, starting from each grid node  ──── #
# --------------------------------------------------------------------------- #
def fixed_points(N, M, U, V, *, params, tol=1e-3):
    """Use every grid node as an initial guess for a root of (Δn,Δm).

    Return unique roots within *tol*.

    Args:
        N: Meshgrid of environment values.
        M: Meshgrid of mean action values.
        U: Flow vector component Δn at each grid node.
        V: Flow vector component Δm at each grid node.
        params: Dictionary of micro-parameters for the ABM.
        tol: Tolerance for considering two roots as the same.

    Returns:
        roots: Array of unique roots (n, m) where the flow is zero.
        Each root is a point where the flow vector (Δn, Δm) is close to zero.
    """
    roots = []
    for n0, m0 in zip(N.ravel(), M.ravel(), strict=False):
        try:
            root, info, ier, _ = opt.fsolve(  # type: ignore
                lambda x: flow_at_point(*x, params=params, reps=4),
                x0=(n0, m0),
                full_output=True,
            )
            if ier == 1 and not any(np.linalg.norm(root - r) < tol for r in roots):
                roots.append(root)
        except ValueError:
            pass
    return np.array(roots)


# --------------------------------------------------------------------------- #
# 4.  ── main driver that makes the figure  ───────────────────────────────── #
# --------------------------------------------------------------------------- #
def plot_phase_portrait_like_paper(
    *,
    num_grid=26,
    reps_per_node=3,
    savedir="phase_portraits",
    params_override=None,
):
    """Plot the phase portrait of the ABM model, similar to the paper.

    Args:
        num_grid: Number of nodes along each axis of the grid.
        reps_per_node: Number of independent restarts for each grid node.
        savedir: Directory to save the figure.
        params_override: Optional dictionary to override default parameters.

    Returns:
        None. Saves the figure to the specified directory.
    """
    # -------- micro-parameters ------------------------------------------------
    params = dict(
        width=50,
        height=50,
        memory_count=20,
        rationality=0.9,
        gamma_s=0.01,
        env_update_fn=piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01),
    )
    if params_override:
        params.update(params_override)
    # -------- field -----------------------------------------------------------
    N, M, U, V = compute_field(params, grid_N=num_grid, reps=reps_per_node)
    # -------- normalise for nicer arrows -------------------------------------
    speed = np.hypot(U, V)
    U_n = U / (speed + 1e-9)
    V_n = V / (speed + 1e-9)
    # -------- plotting --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.streamplot(
        N,
        M,
        U_n,
        V_n,
        density=1.2,
        linewidth=0.8,
        arrowsize=1.3,
        color="#2166ac",
    )
    # dashed diagonal m = 2n − 1
    ax.plot([0, 1], [-1, 1], ls="--", lw=0.8, color="grey", alpha=0.7)
    # -------- fixed points ----------------------------------------------------
    roots = fixed_points(N, M, U, V, params=params)
    if roots.size:
        ax.scatter(
            roots[:, 0],
            roots[:, 1],
            s=60,
            facecolor="red",
            edgecolor="k",
            zorder=5,
        )
    # -------- aesthetics ------------------------------------------------------
    ax.set(
        xlim=(0, 1),
        ylim=(-1, 1),
        xlabel=r"Environment $(n)$",
        ylabel=r"Mean action $(m)$",
    )
    ax.set_aspect("auto")
    ax.set_title("Phase portrait - ABM")
    ax.grid(ls=":", alpha=0.4)

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    fig.savefig(savedir / "phase_portrait_abm.png", dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    plot_phase_portrait_like_paper()

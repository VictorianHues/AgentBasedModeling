import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from abm_project.mean_field import (
    f_dm_dt,
    f_dn_dt,
    fixedpoint_mean_action,
    solve,
    solve_for_equilibria,
)
from abm_project.plotting import plot_phase_portrait
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def plot_steady_state_expected_action_bifurcation():
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    for s in (1.8, 2, 2.2):
        cs = np.linspace(0, 1, 1000)
        solutions = []
        unstable = []
        for c in cs:
            roots = fixedpoint_mean_action(
                s=s, c=c, rationality=1, ignore_warnings=False
            )
            for r in roots.stable():
                solutions.append((c, r))
            for r in roots.unstable():
                unstable.append((c, r))

        solutions = np.asarray(solutions).T
        unstable = np.asarray(unstable).T
        ax.scatter(solutions[0], solutions[1], s=1, label=f"{s=}")
        if unstable.size:
            ax.scatter(unstable[0], unstable[1], s=1, label=f"{s=} (unstable)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Strength of social norms")
    ax.set_ylabel(r"Steady-state $\bar{a}$")
    ax.legend(
        title="Willingness to cooperate",
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0, 0, 0),
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig("bifurcation_action.png", dpi=500)
    plt.show()


# def plot_steady_state_probability_bifurcation():
#    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
#    for s in (1.95, 2, 2.05):
#        cs = np.linspace(0, 1, 1000)
#        solutions = []
#        for c in cs:
#            roots = solve_for_fixedpoint_pc(s=s, b0=1 - c, b1=c, ignore_warnings=True)
#            for r in roots:
#                solutions.append((c, r))
#
#        solutions = np.asarray(solutions).T
#        ax.scatter(solutions[0], solutions[1], s=1, label=f"{s=}")
#    ax.set_xlim(0, 1)
#    ax.set_ylim(0, 1)
#    ax.set_xlabel("Strength of social norms")
#    ax.set_ylabel("Steady-state $P(C)$")
#    ax.legend(
#        title="Willingness to cooperate",
#        ncols=3,
#        loc="lower center",
#        bbox_to_anchor=(0.5, 1.0, 0, 0),
#        frameon=False,
#    )
#    ax.spines["top"].set_visible(False)
#    ax.spines["right"].set_visible(False)
#    fig.savefig("bifurcation.png", dpi=500)
#    plt.show()


# def plot_dndt():
#    s = 3
#    c = 0.8
#    b = 1 - c
#
#    recovery = 1
#    pollution = 2
#
#    n_update_rate = 0.01
#
#    n_prime = f_dn_dt(recovery, pollution, n_update_rate)
#    roots = solve_for_fixedpoint_pc(s, b, c)
#
#    ns = np.linspace(0, 1, 1000)
#    for pc in roots:
#        results = []
#        for n in ns:
#            derivative = n_prime(n, pc)
#            results.append((n, derivative))
#        results = np.asarray(results).T
#
#        plt.scatter(results[0], results[1], label=f"$p_c = {pc:.2f}$", s=1)
#    plt.xlabel("$n$")
#    plt.ylabel(r"$\frac{dn}{dt}$", rotation=0)
#    plt.legend()
#
#    plt.show()


def plot_hysteresis():
    cs = (0.5, 0.75)
    linecolors = ("#1f77b4", "#ff7f0e")
    supports = np.linspace(0, 4, 1000)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    for c, linecolor in zip(cs, linecolors, strict=True):
        stable_lower = []
        stable_upper = []
        unstable = []
        for s in supports:
            roots = fixedpoint_mean_action(s=s, c=c, ignore_warnings=False)
            if roots.lower:
                stable_lower.append((s, roots.lower))
            if roots.upper:
                stable_upper.append((s, roots.upper))
            if roots.middle:
                unstable.append((s, roots.middle))

        stable_lower = np.asarray(stable_lower).T
        stable_upper = np.asarray(stable_upper).T
        unstable = np.asarray(unstable).T
        if not unstable.size:
            stable = np.hstack((stable_lower, stable_upper))
            ax.plot(stable[0], stable[1], color=linecolor, label=f"$c={c}$")
        else:
            ax.plot(stable_lower[0], stable_lower[1], color=linecolor, label=f"$c={c}$")
            ax.plot(stable_upper[0], stable_upper[1], color=linecolor)
            ax.plot(unstable[0], unstable[1], color=linecolor, linestyle="dashed")

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$", rotation=0)
    ax.set_xlim(0, 4)
    ax.set_ylim(-1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0, 0, 0), frameon=False
    )
    plt.show()


def phase_portrait_n_m(
    c: float,
    recovery: float,
    pollution: float,
    rationality: float = 1,
    n_update_rate: float = 0.01,
    s_update_rate: float = 0.001,
    plot_dm_dt_nullcline: bool = False,
    ax: Axes | None = None,
):
    b = 1 - c
    ALPHA = 1
    BETA = 1

    min_m = fixedpoint_mean_action(0, c, rationality).lower + 1e-3
    max_m = fixedpoint_mean_action(4, c, rationality).upper - 1e-3

    ns = np.linspace(0, 1, 60)
    ms = np.linspace(min_m, max_m, 60)

    m_prime = f_dm_dt(rationality, b, c, ALPHA, BETA, rate=s_update_rate)
    n_prime = f_dn_dt(recovery, pollution, rate=n_update_rate)

    # Calculate derivatives for mean action
    N, M = np.meshgrid(ns, ms)
    dm_dt = m_prime(M, N)

    # Calculate derivatives for environment
    pc = (ms + 1) / 2
    N, P = np.meshgrid(ns, pc)
    dn_dt = n_prime(N, P)

    # Normalise for plotting
    DN_DT = dn_dt
    DM_DT = dm_dt

    # fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if not ax:
        ax = plt.gca()
    ax.streamplot(N, M, DN_DT, DM_DT, density=[1.5, 2], linewidth=0.5)

    # Draw nullclines
    # m' = 0 when s' = 0
    if plot_dm_dt_nullcline:
        sigma_n = 4 * ns * (1 - ns)
        s = 4 * ALPHA * sigma_n / (BETA * (1 - sigma_n) + ALPHA * sigma_n)
        da_dt_nullcline_ns = []
        da_dt_nullcline_ms = []
        for n, _s in zip(ns, s, strict=True):
            roots = fixedpoint_mean_action(_s, c, rationality)
            for m in roots.stable():
                da_dt_nullcline_ns.append(n)
                da_dt_nullcline_ms.append(m)
        ax.scatter(da_dt_nullcline_ns, da_dt_nullcline_ms, s=10, color="red")

        da_dt_nullcline_ns = []
        da_dt_nullcline_ms = []
        for n, _s in zip(ns, s, strict=True):
            roots = fixedpoint_mean_action(_s, c, rationality)
            for m in roots.unstable():
                da_dt_nullcline_ns.append(n)
                da_dt_nullcline_ms.append(m)
        ax.scatter(da_dt_nullcline_ns, da_dt_nullcline_ms, s=10, color="green")

    # n' = 0
    stationary_ns = (recovery * (1 + ms)) / (pollution * (1 - ms) + recovery * (1 + ms))

    ax.plot(
        stationary_ns,
        ms,
        linestyle="dashed",
        color="black",
        linewidth=0.7,
        label=r"$\frac{dn}{dt} = 0$",
    )

    eq_n, eq_m = solve_for_equilibria(
        b=b, c=c, rationality=rationality, recovery=recovery, pollution=pollution
    )

    ax.scatter(eq_n, eq_m, color="red")

    ax.set_xlabel("Environment ($n$)")
    ax.set_ylabel("Mean action ($m$)")
    plt.show()


# def phaseplot_n_s():
#    ns = np.linspace(0, 1, 100)
#    supports = np.linspace(0, 4, 100)
#    c = 0.5
#    b = 1 - c
#    alpha = 1
#    beta = 1
#    recovery = 1
#    pollution = 1
#    n_update_rate = 0.01
#    s_update_rate = 0.001
#
#    s_prime = f_ds_dt(alpha, beta, rate=s_update_rate)
#    n_prime = f_dn_dt(recovery, pollution, rate=n_update_rate)
#
#    # Calculate derivatives for environment
#    N, S = np.meshgrid(ns, supports)
#    ds_dt = s_prime(N, S)
#
#    # Calculate derivatives for support (willingness to cooperate)
#    roots = [solve_for_fixedpoint_pc(s, b, c) for s in supports]
#    pc = [
#        min(s_roots) if s <= 2 else max(s_roots)
#        for (s, s_roots) in zip(supports, roots, strict=True)
#    ]
#    N, P = np.meshgrid(ns, pc)
#    dn_dt = n_prime(N, P)
#
#    # Normalise for plotting
#    grad_length = np.sqrt(dn_dt**2 + ds_dt**2)
#    DN_DT = dn_dt / grad_length
#    DS_DT = ds_dt / grad_length
#    M = np.hypot(dn_dt, ds_dt)  # Vector length for colour
#
#    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
#    ax.quiver(N, S, DN_DT, DS_DT, M, width=0.002, scale=220)
#    ax.set_xlabel("Environment ($n$)")
#    ax.set_ylabel("Willingness to cooperate ($s$)")
#    plt.show()


def plot_trajectory():
    c = 0.5
    b = 1 - c
    alpha = 1
    beta = 1
    recovery = 1
    pollution = 1

    n_update_rate = 0.01
    s_update_rate = 0.001

    n0 = 1
    s0 = 0
    num_steps = 2000

    t, (n, s, sp, a, _) = solve(
        b,
        c,
        alpha,
        beta,
        pollution,
        recovery,
        n_update_rate,
        s_update_rate,
        n0,
        s0,
        num_steps,
    )

    fig, axes = plt.subplots(
        nrows=4, figsize=(6, 8), constrained_layout=True, sharex=True
    )

    axes[0].plot(t, n)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(0, num_steps)
    axes[0].set_ylabel("Environment")

    axes[1].plot(t, s)
    axes[1].set_ylim(0, 4)
    axes[1].set_ylabel("Support")

    axes[2].plot(t, sp)
    axes[2].set_ylim(0, 4)
    axes[2].set_ylabel("Social pressure")

    axes[3].plot(t, a)
    axes[3].set_ylim(-1, 1)
    axes[3].set_ylabel("Action")

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Time")

    plt.show()


def example_phase_portrait():
    c = 0.5
    recovery = 1
    pollution = 1
    rationality = 1

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_phase_portrait(
        c=c,
        recovery=recovery,
        pollution=pollution,
        rationality=rationality,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"Environment ($n$)")
    ax.set_ylabel(r"Mean action ($m$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


def example_phase_portrait_with_model_trajectory():
    width = 100
    height = 100
    num_steps = 1000
    c = 0.5
    recovery = 1
    pollution = 1
    rationality = 3

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plot_phase_portrait(
        c=c,
        recovery=recovery,
        pollution=pollution,
        rationality=rationality,
    )

    # Plot model trajectories through phase portrait
    env = piecewise_exponential_update(
        recovery=recovery, pollution=pollution, gamma=0.01
    )
    for i in range(3):
        rng = np.random.default_rng(i)
        model = VectorisedModel(
            num_agents=width * height,
            width=width,
            height=height,
            rng=rng,
            env_update_fn=env,
            rationality=rationality,
            max_storage=num_steps,
        )
        model.run(num_steps)
        n = model.environment[: model.time].mean(axis=1)
        m = model.action[: model.time].mean(axis=1)
        ax.plot(n, m, label="Simulation")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r"Environment ($n$)")
    ax.set_ylabel(r"Mean action ($m$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


def main():
    example_phase_portrait()
    example_phase_portrait_with_model_trajectory()
    plot_hysteresis()

    # plot_steady_state_expected_action_bifurcation()


if __name__ == "__main__":
    main()

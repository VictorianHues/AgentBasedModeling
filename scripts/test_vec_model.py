import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from abm_project.mean_field import solve
from abm_project.cluster_analysis import cluster_time_series
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def plot_abm_vs_meanfield_time_series(savedir: Path | None = None):
    savedir = savedir or Path(".")

    rationality = 2
    num_agents = 900
    width = 30
    height = 30
    num_steps = 3000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    gamma_s = 0.005
    rng = None

    repeats = 30

    fig, axes = plt.subplots(
        nrows=4,
        figsize=(6, 10),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    environment = np.empty((repeats, num_steps + 1))
    action = np.empty((repeats, num_steps + 1))
    social_pressure = np.empty((repeats, num_steps + 1))
    support = np.empty((repeats, num_steps + 1))
    for r in range(repeats):
        model = VectorisedModel(
            num_agents=num_agents,
            width=width,
            height=height,
            memory_count=memory_count,
            rng=rng,
            env_update_fn=env_update_fn,
            rationality=rationality,
            neighb_prediction_option=None,
            severity_benefit_option="adaptive",
            max_storage=num_steps,
            moore=False,
            gamma_s=gamma_s,
        )
        model.run(num_steps)
        environment[r] = model.environment.mean(axis=1)
        action[r] = model.action.mean(axis=1)
        support[r] = model.s.mean(axis=1)
        mean_local_action = (model.adj @ model.action.T).T
        social_pressure[r] = (
            model.b[1] * (model.action - mean_local_action) ** 2
        ).mean(axis=1)
    t = np.arange(num_steps + 1)

    # Plot mean environment
    n_mean = environment.mean(axis=0)
    n_std = environment.std(axis=0, ddof=1)
    n_ci = 1.97 * n_std / np.sqrt(repeats)
    axes[0].plot(t, n_mean)
    axes[0].fill_between(t, n_mean - n_ci, n_mean + n_ci, alpha=0.3)

    # Plot mean support
    s_mean = support.mean(axis=0)
    s_std = support.std(axis=0, ddof=1)
    s_ci = 1.97 * s_std / np.sqrt(repeats)
    axes[1].plot(t, s_mean)
    axes[1].fill_between(t, s_mean - s_ci, s_mean + s_ci, alpha=0.3)

    # Plot mean social pressure
    p_mean = social_pressure.mean(axis=0)
    p_std = social_pressure.std(axis=0, ddof=1)
    p_ci = 1.97 * p_std / np.sqrt(repeats)
    axes[2].plot(t, p_mean)
    axes[2].fill_between(t, p_mean - p_ci, p_mean + p_ci, alpha=0.3)

    # Plot mean action
    a_mean = action.mean(axis=0)
    a_std = action.std(axis=0, ddof=1)
    a_ci = 1.97 * a_std / np.sqrt(repeats)
    axes[3].plot(t, a_mean)
    axes[3].fill_between(t, a_mean - a_ci, a_mean + a_ci, alpha=0.3)

    t, (n, s, sp, a, _) = solve(
        b=0.5,
        c=0.5,
        recovery=1,
        pollution=1,
        alpha=1,
        beta=1,
        n_update_rate=0.01,
        s_update_rate=gamma_s,
        n0=1,
        m0=-0.93,
        num_steps=num_steps,
    )

    axes[0].plot(t, n, linestyle="dashed", linewidth=1, color="black")
    axes[1].plot(t, s, linestyle="dashed", linewidth=1, color="black")
    axes[2].plot(t, sp, linestyle="dashed", linewidth=1, color="black")
    axes[3].plot(t, a, linestyle="dashed", linewidth=1, color="black")

    axes[0].set_ylabel("Environment ($n$)")
    axes[1].set_ylabel("Support ($s$)")
    axes[2].set_ylabel("Social pressure ($p$)")
    axes[3].set_ylabel("Action ($a$)")

    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 4)
    axes[2].set_ylim(0, 4)
    axes[3].set_ylim(-1, 1)

    for ax in axes.flatten():
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Time")

    fig.savefig(
        savedir / "system_time_series_mean_field.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_environment_for_varying_rationality(savedir: Path | None = None):
    savedir = savedir or Path(".")

    num_agents = 900
    width = 30
    height = 30
    num_steps = 1000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = None

    repeats = 30
    rationality = np.array([1, 2, 3])

    fig, axes = plt.subplots(
        nrows=4,
        figsize=(6, 10),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    for lmbda in rationality:
        environment = np.empty((repeats, num_steps + 1))
        action = np.empty((repeats, num_steps + 1))
        social_pressure = np.empty((repeats, num_steps + 1))
        support = np.empty((repeats, num_steps + 1))
        for r in range(repeats):
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=lmbda,
                simmer_time=1,
                neighb_prediction_option=None,
                severity_benefit_option=None,
                max_storage=num_steps,
            )
            model.run(num_steps)
            environment[r] = model.environment.mean(axis=1)
            action[r] = model.action.mean(axis=1)
            support[r] = model.s.mean(axis=1)
            mean_local_action = (model.adj @ model.action.T).T
            social_pressure[r] = (
                model.b[1] * (model.action - mean_local_action) ** 2
            ).mean(axis=1)
        t = np.arange(num_steps + 1)

        # Plot mean environment
        n_mean = environment.mean(axis=0)
        n_std = environment.std(axis=0, ddof=1)
        n_ci = 1.97 * n_std / np.sqrt(repeats)
        axes[0].plot(t, n_mean, label=f"$\\lambda = {lmbda:.2f}$")
        axes[0].fill_between(t, n_mean - n_ci, n_mean + n_ci, alpha=0.3)

        # Plot mean support
        s_mean = support.mean(axis=0)
        s_std = support.std(axis=0, ddof=1)
        s_ci = 1.97 * s_std / np.sqrt(repeats)
        axes[1].plot(t, s_mean, label=f"$\\lambda = {lmbda:.2f}$")
        axes[1].fill_between(t, s_mean - s_ci, s_mean + s_ci, alpha=0.3)

        # Plot mean social pressure
        p_mean = social_pressure.mean(axis=0)
        p_std = social_pressure.std(axis=0, ddof=1)
        p_ci = 1.97 * p_std / np.sqrt(repeats)
        axes[2].plot(t, p_mean, label=f"$\\lambda = {lmbda:.2f}$")
        axes[2].fill_between(t, p_mean - p_ci, p_mean + p_ci, alpha=0.3)

        # Plot mean action
        a_mean = action.mean(axis=0)
        a_std = action.std(axis=0, ddof=1)
        a_ci = 1.97 * a_std / np.sqrt(repeats)
        axes[3].plot(t, a_mean, label=f"$\\lambda = {lmbda:.2f}$")
        axes[3].fill_between(t, a_mean - a_ci, a_mean + a_ci, alpha=0.3)

    axes[0].set_ylabel("Environment ($n$)")
    axes[1].set_ylabel("Support ($s$)")
    axes[2].set_ylabel("Social pressure ($p$)")
    axes[3].set_ylabel("Action ($a$)")

    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 4)
    axes[2].set_ylim(0, 4)
    axes[3].set_ylim(-1, 1)

    for ax in axes.flatten():
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Time")

    fig.savefig(
        savedir / "system_time_series_varying_rationality.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_steady_state_environment_for_varying_rationality(savedir: Path | None = None):
    savedir = savedir or Path(".")

    num_agents = 900
    width = 30
    height = 30
    num_steps = 3000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = None

    repeats = 20
    rationality = np.linspace(0, 4, 30)

    results = np.empty((repeats, len(rationality)))
    for r in range(repeats):
        for i, lmbda in enumerate(rationality):
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=lmbda,
                simmer_time=1,
                neighb_prediction_option=None,
                severity_benefit_option=None,
                max_storage=num_steps,
            )
            model.run(num_steps)
            results[r, i] = model.environment[model.time].mean()

    # Plot mean environmen at steady state
    fig, ax = plt.subplots(
        figsize=(6, 4),
        constrained_layout=True,
    )

    mean = results.mean(axis=0)
    std = results.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)
    ax.plot(rationality, mean)
    ax.fill_between(rationality, mean - ci, mean + ci, alpha=0.3)

    ax.set_xlabel(r"Rationality ($\lambda$)")
    ax.set_ylabel(r"Equilibrium mean environment ($\overline{n}^*$)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(
        savedir / "equilibrium_env_vs_rationality.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_environment_for_varying_pessimism(savedir: Path | None = None):
    savedir = savedir or Path(".")

    rationality = 1
    simmer_time = 10
    num_agents = 900
    width = 30
    height = 30
    num_steps = 1000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = None

    repeats = 15
    prop_pessimistic = np.array([0.1, 0.5, 0.9])
    pessimism_levels = np.array([1.5, 3.0])

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(12, 10),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    for i, pess_level in enumerate(pessimism_levels):
        for pess in prop_pessimistic:
            environment = np.empty((repeats, num_steps + 1))
            action = np.empty((repeats, num_steps + 1))
            social_pressure = np.empty((repeats, num_steps + 1))
            support = np.empty((repeats, num_steps + 1))
            for r in range(repeats):
                model = VectorisedModel(
                    num_agents=num_agents,
                    width=width,
                    height=height,
                    memory_count=memory_count,
                    rng=rng,
                    env_update_fn=env_update_fn,
                    rationality=rationality,
                    simmer_time=simmer_time,
                    neighb_prediction_option=None,
                    severity_benefit_option=None,
                    max_storage=num_steps,
                    prop_pessimistic=pess,
                    pessimism_level=pess_level,
                )
                model.run(num_steps)
                environment[r] = model.environment.mean(axis=1)
                action[r] = model.action.mean(axis=1)
                support[r] = model.s.mean(axis=1)
                mean_local_action = (model.adj @ model.action.T).T
                social_pressure[r] = (
                    model.b[1] * (model.action - mean_local_action) ** 2
                ).mean(axis=1)
            t = np.arange(num_steps + 1)

            # Plot mean environment
            n_mean = environment.mean(axis=0)
            n_std = environment.std(axis=0, ddof=1)
            n_ci = 1.97 * n_std / np.sqrt(repeats)
            axes[0, i].plot(t, n_mean, label=f"$p = {pess}$")
            axes[0, i].fill_between(t, n_mean - n_ci, n_mean + n_ci, alpha=0.3)

            # Plot mean support
            s_mean = support.mean(axis=0)
            s_std = support.std(axis=0, ddof=1)
            s_ci = 1.97 * s_std / np.sqrt(repeats)
            axes[1, i].plot(t, s_mean, label=f"$p = {pess}$")
            axes[1, i].fill_between(t, s_mean - s_ci, s_mean + s_ci, alpha=0.3)

            # Plot mean social pressure
            p_mean = social_pressure.mean(axis=0)
            p_std = social_pressure.std(axis=0, ddof=1)
            p_ci = 1.97 * p_std / np.sqrt(repeats)
            axes[2, i].plot(t, p_mean, label=f"$p = {pess}$")
            axes[2, i].fill_between(t, p_mean - p_ci, p_mean + p_ci, alpha=0.3)

            # Plot mean action
            a_mean = action.mean(axis=0)
            a_std = action.std(axis=0, ddof=1)
            a_ci = 1.97 * a_std / np.sqrt(repeats)
            axes[3, i].plot(t, a_mean, label=f"$p = {pess}$")
            axes[3, i].fill_between(t, a_mean - a_ci, a_mean + a_ci, alpha=0.3)

        axes[0, i].set_ylabel("Environment ($n$)")
        axes[1, i].set_ylabel("Support ($s$)")
        axes[2, i].set_ylabel("Social pressure ($p$)")
        axes[3, i].set_ylabel("Action ($a$)")

        axes[0, i].set_ylim(0, 1)
        axes[1, i].set_ylim(0, 4)
        axes[2, i].set_ylim(0, 4)
        axes[3, i].set_ylim(-1, 1)

    axes[0, 0].set_title(f"Pessimism level: {pessimism_levels[0]:.2f}")
    axes[0, 1].set_title(f"Pessimism level: {pessimism_levels[1]:.2f}")

    for ax in axes.flatten():
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Time")

    fig.savefig(
        savedir / "system_time_series_varying_pessimism.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_cluster_time_series(
    savedir: Path | None = None,
    model: VectorisedModel | None = None,
    option: str = "action",
):
    savedir = savedir or Path(".")

    nc, c1 = cluster_time_series(model, option=option)
    print(f"Number of clusters at each time step: {nc}")
    print(f"Largest cluster fraction at each time step: {c1 / model.num_agents}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(nc, label=f"Number of clusters {option}")
    ax.plot(c1 / (model.num_agents), label="Largest cluster")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Number of Clusters Over Time")
    ax.legend()

    fig.savefig(
        savedir / f"cluster_analysis_{option}_mem_count_{model.memory_count}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def main():
    results_dir = Path("vectorised_model_results")
    results_dir.mkdir(exist_ok=True)

    # plot_support_derivative(savedir=results_dir)
    plot_abm_vs_meanfield_time_series(savedir=results_dir)
    plot_environment_for_varying_rationality(savedir=results_dir)
    plot_environment_for_varying_pessimism(savedir=results_dir)
    plot_steady_state_environment_for_varying_rationality(savedir=results_dir)

    num_agents = 2500
    width = 50
    height = 50
    num_steps = 1000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = None

    start = time.time()
    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=1.8,
        simmer_time=1,
        neighb_prediction_option="logistic",
        severity_benefit_option=None,
        max_storage=num_steps,
    )
    model.run(num_steps)
    end = time.time()

    print(f"Model state after {num_steps} steps ({end - start:.2f}s).")
    print("======================================")
    print("Environment:")
    print(f"Mean: {model.environment[num_steps].mean():.2f}")
    print(f"Min: {model.environment[num_steps].min():.2f}")
    print(f"Max: {model.environment[num_steps].max():.2f}")
    print()
    print("Support for mitigation:")
    print(f"Mean: {model.s[num_steps].mean():.2f}")
    print(f"Min: {model.s[num_steps].min():.2f}")
    print(f"Max: {model.s[num_steps].max():.2f}")

    env_status_history = model.environment[: model.time + 1]
    env_status_history = env_status_history.reshape((-1, model.height, model.width))
    num_steps = env_status_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(env_status_history[0], cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
    ax.set_title("Agent Environment Status Over Time")

    def update(frame):
        im.set_array(env_status_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, blit=True, interval=100, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Environment Status")
    plt.tight_layout()
    ani.save(results_dir / "example_env.mp4", dpi=150)

    # Animate actions
    agent_action_history = model.action[: model.time + 1]
    agent_action_history = agent_action_history.reshape((-1, model.height, model.width))
    num_steps = agent_action_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(
        agent_action_history[0], cmap=ListedColormap(["red", "green"]), origin="lower"
    )
    ax.set_title("Agent Actions Over Time")

    def update(frame):
        im.set_array(agent_action_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, blit=True, interval=100, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Agent Action (-1 or 1)")
    plt.tight_layout()
    ani.save(results_dir / "example_actions.mp4", dpi=150)

    plot_cluster_time_series(savedir=results_dir, model=model)


if __name__ == "__main__":
    main()

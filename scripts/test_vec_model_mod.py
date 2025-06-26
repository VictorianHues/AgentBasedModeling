import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def plot_environment_for_varying_rationality(savedir: Path | None = None):
    savedir = savedir or Path(".")

    num_agents = 900
    width = 30
    height = 30
    num_steps = 1000
    memory_count = 1
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None
    neighb_prediction_option = None  #  "logistic", "linear" or None
    severity_benefit_option = "adaptive"  # "adaptive" or None
    radius_option = "all"  # "single" or "all"

    repeats = 30
    rationality = np.array([1, 2, 3])

    fig, axes = plt.subplots(
        nrows=4,
        figsize=(7, 4),
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
                neighb_prediction_option=neighb_prediction_option,
                severity_benefit_option=severity_benefit_option,
                radius_option=radius_option,
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
    num_steps = 1000
    memory_count = 10
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None

    repeats = 50
    rationality = np.linspace(0, 1, 30)

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
                neighb_prediction_option="linear",
                severity_benefit_option="adaptive",
                radius_option="single",
                max_storage=num_steps,
                b_1=np.full(num_agents, 1.0, dtype=np.float64),
                b_2=np.full(num_agents, 1.0, dtype=np.float64),
            )
            model.run(num_steps)
            results[r, i] = model.environment[model.time].mean()

    # Plot mean environment at steady state
    fig, ax = plt.subplots(
        figsize=(7, 4),
        constrained_layout=True,
    )

    mean = results.mean(axis=0)
    std = results.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)
    ax.plot(rationality, mean, label="Mean")
    ax.fill_between(rationality, mean - ci, mean + ci, alpha=0.3)

    # maximum absolute derivative
    dmean = np.gradient(mean, rationality)
    idx_steepest = np.argmax(np.abs(dmean))
    ax.plot(rationality[idx_steepest], mean[idx_steepest], "ro", label="Steepest point")
    ax.annotate(
        f"Steepest\n($\\lambda$={rationality[idx_steepest]:.2f})",
        xy=(rationality[idx_steepest], mean[idx_steepest]),
        xytext=(rationality[idx_steepest] + 0.2, mean[idx_steepest]),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=10,
        ha="left",
    )

    ax.set_xlabel(r"Rationality ($\lambda$)")
    ax.set_ylabel(r"Equilibrium mean environment ($\overline{n}^*$)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    fig.savefig(
        savedir / "equilibrium_env_vs_rationality.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    results_dir = Path("vectorised_model_results")
    results_dir.mkdir(exist_ok=True)

    # plot_environment_for_varying_rationality(savedir=results_dir)
    # plot_steady_state_environment_for_varying_rationality(savedir=results_dir)

    num_agents = 2500
    width = 50
    height = 50
    num_steps = 3000
    memory_count = 10
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None
    rationality = 0.41
    simmer_time = 1
    neighb_prediction_option = "linear"  #  "logistic", "linear" or None
    severity_benefit_option = "adaptive"  # "adaptive" or None
    radius_option = "single"  # "single" or "all"

    start = time.time()
    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality,
        simmer_time=simmer_time,
        neighb_prediction_option=neighb_prediction_option,
        severity_benefit_option=severity_benefit_option,
        radius_option=radius_option,
        max_storage=num_steps,
        b_1=np.full(num_agents, 1.0, dtype=np.float64),
        b_2=np.full(num_agents, 1.0, dtype=np.float64),
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

    # Take every 5th frame for environment status animation
    env_status_history = model.environment[: model.time + 1]
    env_status_history = env_status_history.reshape((-1, model.height, model.width))
    env_status_history = env_status_history[::5]
    num_env_frames = env_status_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(env_status_history[0], cmap="RdYlGn", origin="lower", vmin=0, vmax=1)
    ax.set_title("Agent Environment Status Over Time")

    def update_env(frame):
        im.set_array(env_status_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update_env, frames=num_env_frames, blit=True, interval=100, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Environment Status")
    plt.tight_layout()
    ani.save(results_dir / "example_env.gif", dpi=150)
    ani.save(results_dir / "example_env.mp4", dpi=150, writer="ffmpeg")

    # Take every 5th frame for agent actions animation
    agent_action_history = model.action[: model.time + 1]
    agent_action_history = agent_action_history.reshape((-1, model.height, model.width))
    agent_action_history = agent_action_history[::5]
    num_action_frames = agent_action_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(
        agent_action_history[0], cmap=ListedColormap(["red", "green"]), origin="lower"
    )
    ax.set_title("Agent Actions Over Time")

    def update_action(frame):
        im.set_array(agent_action_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update_action,
        frames=num_action_frames,
        blit=True,
        interval=100,
        repeat=False,
    )

    plt.colorbar(im, ax=ax, label="Agent Action (-1 or 1)")
    plt.tight_layout()
    ani.save(results_dir / "example_actions.gif", dpi=150)
    ani.save(results_dir / "example_actions.mp4", dpi=150, writer="ffmpeg")


if __name__ == "__main__":
    main()
    # plot_mean_env_for_varying_attribute()

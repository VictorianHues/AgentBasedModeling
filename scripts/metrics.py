import matplotlib.pyplot as plt
import numpy as np

from abm_project.metrics import (
    action_variance_over_time,
    cumulative_environmental_harm,
    environment_variance_over_time,
    expected_action_volatility,
    expected_environment_volatility,
    extract_actions,
    extract_environment,
    pluralistic_ignorance,
    probability_repeated_action,
)
from abm_project.oop_model import BaseModel


def main():
    """Main function to run the agent-based model simulation."""
    num_agents = 250
    width = 25
    height = 10
    radius = 1
    num_steps = 1000
    memory_count = 1
    env_update_option = "linear"
    rng = None

    def env_status_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    def peer_pressure_coeff_fn():
        if rng:
            return rng.uniform(0, 1)
        else:
            return np.random.uniform(0.0, 1.0)

    def env_perception_coeff_fn():
        if rng:
            return rng.uniform(0.0, 1.0)
        else:
            return np.random.uniform(0.0, 1.0)

    model = BaseModel(
        num_agents=num_agents,
        width=width,
        height=height,
        radius=radius,
        memory_count=memory_count,
        rng=rng,
        env_update_option=env_update_option,
        env_status_fn=env_status_fn,
        peer_pressure_coeff_fn=peer_pressure_coeff_fn,
        env_perception_coeff_fn=env_perception_coeff_fn,
    )

    model.run(num_steps)

    p_actual, p_ideal, n, a = pluralistic_ignorance(model)
    n = n[:100]
    a = a[:100]
    p_actual = p_actual.mean(axis=0)[:100]
    p_ideal = p_ideal.mean(axis=0)[:100]
    fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    t = np.arange(n.size)
    ax.plot(t, p_actual, label=r"$p_\text{actual}$")
    ax.plot(t, p_ideal, label=r"$p_\text{ideal}$")
    ax.plot(t, n, label=r"Mean environment")
    ax.set_xlabel("Time")
    fig.legend(loc="outside center right")

    plt.show()

    k = 20
    p_cooperate, peer_pressure = probability_repeated_action(model, k=k)
    dp_cooperate = p_cooperate - p_cooperate[:, 0][:, None]

    fig, axes = plt.subplots(
        ncols=5, figsize=(10, 3), sharey=True, constrained_layout=True
    )
    t = np.arange(k + 1)
    for lb, ub, ax in zip(
        (0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.0), axes.flatten(), strict=True
    ):
        p = dp_cooperate[(lb < peer_pressure) & (peer_pressure <= ub)]
        avg = np.median(p, axis=0)
        ax.plot(t, avg)
        try:
            lower = np.percentile(p, 25, axis=0)
            upper = np.percentile(p, 75, axis=0)
            ax.fill_between(t, lower, upper, alpha=0.3)
            ax.set_title(f"${lb} < c \\leq {ub}$")
        except Exception:
            continue

    fig.supylabel(r"$\frac{dp_c}{dt}$")
    fig.supxlabel(r"Time since first cooperation")
    fig.suptitle("Change in cooperation probability after first cooperation")
    plt.show()

    μ_a, σ2_a = action_variance_over_time(model)
    μ_n, σ2_n = environment_variance_over_time(model)
    t = np.arange(μ_a.size)
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5), constrained_layout=True)
    axes[0].plot(t, μ_a, label=r"$\mathbb{E}[a]$")
    axes[0].plot(t, μ_n, label=r"$\mathbb{E}[n]$")
    axes[0].fill_between(t, μ_a - np.sqrt(σ2_a), μ_a + np.sqrt(σ2_a), alpha=0.3)
    axes[0].fill_between(t, μ_n - np.sqrt(σ2_n), μ_n + np.sqrt(σ2_n), alpha=0.3)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Mean action and environment over time")
    axes[0].legend()
    axes[1].plot(t, σ2_a, label=r"$σ^2 (a_t)$")
    axes[1].plot(t, σ2_n, label=r"$σ^2 (n_t)$")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Variance")
    axes[1].legend()

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.show()

    cumulative_harm = cumulative_environmental_harm(model)
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.plot(np.arange(cumulative_harm.size), cumulative_harm)
    ax.set_xlabel("Time")
    ax.set_ylabel("Net degradation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Cumulative environmental damage over time")
    plt.show()

    n = extract_environment(model).mean(axis=0)
    a = (extract_actions(model) / 2 + 0.5).mean(axis=0)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(n, a)
    plt.show()

    repeats = 5
    peer_pressure_level = np.arange(1, 13)
    action_volatility = np.empty((repeats, peer_pressure_level.size))
    env_volatility = np.empty((repeats, peer_pressure_level.size))
    num_steps = 100
    for r in range(repeats):
        for i, a in enumerate(peer_pressure_level):

            def peer_pressure_coeff_fn():
                return np.random.beta(a=a, b=1)  # noqa

            model = BaseModel(
                num_agents=num_agents,
                width=width,
                height=height,
                radius=radius,
                memory_count=memory_count,
                rng=rng,
                env_update_option=env_update_option,
                env_status_fn=env_status_fn,
                peer_pressure_coeff_fn=peer_pressure_coeff_fn,
                env_perception_coeff_fn=env_perception_coeff_fn,
            )

            model.run(num_steps)

            action_volatility[r, i] = expected_action_volatility(model)
            env_volatility[r, i] = expected_environment_volatility(model)

    fig, axes = plt.subplots(ncols=2, figsize=(5, 3), constrained_layout=True)
    axes[0].plot(peer_pressure_level, action_volatility.mean(axis=0))
    axes[1].plot(peer_pressure_level, env_volatility.mean(axis=0))
    axes[0].set_ylabel("Expected action volatility")
    axes[1].set_ylabel("Expected environment volatility")
    fig.supxlabel("Strength of social norms")
    plt.show()


if __name__ == "__main__":
    main()

"""Run 2D lattice model experiment from Kraan paper."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from abm_project.kraan import KraanModel, exogenous_env


def plot_avg_attitude_vs_env(
    environment: npt.NDArray[np.float64],
    avg_attitude: npt.NDArray[np.float64],
    ax: Axes,
):
    """Line-plot of average attitude versus environment.

    Args:
        environment: 1D numpy array containing state of environment at time of each
            measurement.
        avg_attitude: 2D numpy array (repeats, measurements) containing the average
            action across agents at each timestep, in each repeat of the experiment.
        ax: Matplotlib Axes object to plot graph onto.
    """
    halfway = environment.size // 2
    env_incr = environment[: halfway + 1]
    env_decr = environment[halfway + 1 :]
    avg_attitude_incr = avg_attitude[:, : halfway + 1]
    avg_attitude_decr = avg_attitude[:, halfway + 1 :]

    # Extract median and 25/75% percentiles for plots
    avg_attitude_incr_median = np.median(avg_attitude_incr, axis=0)
    avg_attitude_decr_median = np.median(avg_attitude_decr, axis=0)
    avg_attitude_incr_perc = np.percentile(avg_attitude_incr, [25, 75], axis=0)
    avg_attitude_decr_perc = np.percentile(avg_attitude_decr, [25, 75], axis=0)

    ax.plot(env_incr, avg_attitude_incr_median, label=r"$\frac{dn}{dt} > 0$")
    ax.fill_between(env_incr, *avg_attitude_incr_perc, alpha=0.3)

    ax.plot(env_decr, avg_attitude_decr_median, label=r"$\frac{dn}{dt} < 0$")
    ax.fill_between(env_decr, *avg_attitude_decr_perc, alpha=0.3)


def main():
    """Replicate 2D lattice experiment from Kraan paper."""
    repeats = 30
    update_every = 20
    increment_steps = 40
    decrement_steps = 40
    total_steps = increment_steps + decrement_steps + 1

    # Initialise environment update function
    update_schedule = exogenous_env(
        increment_steps=increment_steps, decrement_steps=decrement_steps
    )

    # Repeat for different neighborhood influence strengths
    cs = (0, 0.5, 1.0)
    fig, axes = plt.subplots(
        ncols=len(cs),
        figsize=(12, 3),
        constrained_layout=True,
        sharey=True,
    )
    for ax, c in zip(axes, cs, strict=False):
        avg_attitude = np.zeros((repeats, total_steps), dtype=np.float64)
        environment = np.zeros(total_steps, dtype=np.float64)

        for r in range(repeats):
            model = KraanModel(
                width=10, height=25, c=c, seed=r, n_update_fn=update_schedule
            )
            for t in model.run(update_steps=total_steps, simmer_steps=update_every):
                environment[t] = model.n
                avg_attitude[r, t] = model.action.mean()

        # Plot results
        plot_avg_attitude_vs_env(environment, avg_attitude, ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(f"$c = {c}$")

    fig.supxlabel("Environment")
    fig.supylabel("Average action")
    ax.legend(title="Median")
    plt.show()


if __name__ == "__main__":
    main()

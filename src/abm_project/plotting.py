"""Plotting functions for agent-based model visualizations."""

import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from tqdm import tqdm

from . import mean_field as mf


def configure_mpl():
    """Configure Matplotlib style."""
    FONT_SIZE_SMALL = 7
    FONT_SIZE_DEFAULT = 9

    plt.rc("font", family="Times New Roman")  # LaTeX default font
    plt.rc("font", weight="normal")
    plt.rc("mathtext", fontset="stix")
    plt.rc("font", size=FONT_SIZE_DEFAULT)
    plt.rc("figure", labelsize=FONT_SIZE_DEFAULT)
    plt.rc("figure", dpi=300)

    sns.set_context(
        "paper",
        rc={
            "axes.linewidth": 0.5,
            "axes.labelsize": FONT_SIZE_DEFAULT,
            "axes.titlesize": FONT_SIZE_DEFAULT,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.4,
            "xtick.labelsize": FONT_SIZE_SMALL,
            "ytick.labelsize": FONT_SIZE_SMALL,
        },
    )


def get_plot_directory(file_name):
    """Get the directory for saving plots.

    Args:
        file_name (str, optional): Name of the file to save the plot. If None
            will return the directory without a file name.

    Returns:
        str: Directory path for saving plots.
    """
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_dir = os.path.join(plot_dir, file_name) if file_name else plot_dir
    return plot_dir


def get_data_directory(file_name):
    """Get the directory for saving data files.

    Args:
        file_name (str, optional): Name of the file to save the data. If None
            will return the directory without a file name.

    Returns:
        str: Directory path for saving data files.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_dir = os.path.join(data_dir, file_name) if file_name else data_dir
    return data_dir


def plot_current_grid_state(
    grid, colormap, title, colorbar_label, file_name=None, clim=None
):
    """Plot the current state of the grid.

    Args:
        grid (np.ndarray): 2D array representing the grid state.
        colormap (str or Colormap): Colormap for the grid visualization.
        title (str): Title of the plot.
        colorbar_label (str): Label for the colorbar.
        file_name (str, optional): Name of the file to save the plot. If None
            will display the plot.
        clim (tuple, optional): Tuple (vmin, vmax) to set colorbar limits.
    """
    im = plt.imshow(grid, cmap=colormap, origin="lower")
    if clim is not None:
        im.set_clim(*clim)
    plt.colorbar(im, label=colorbar_label)
    plt.title(title)
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        print(f"Plot saved to {get_plot_directory(file_name)}")
        plt.close()
    else:
        plt.show()


def animate_grid_states(
    grid_history, colormap, title, colorbar_label, file_name=None, clim=None
):
    """Animate the grid states over time.

    Args:
        grid_history (np.ndarray): 3D array of shape (num_steps, height, width)
        colormap (str or Colormap): Colormap for the grid visualization.
        title (str): Title of the plot.
        colorbar_label (str): Label for the colorbar.
        file_name (str, optional): Name of the file to save the animation.
            If None, will display the plot.
        clim (tuple, optional): Tuple (vmin, vmax) to set colorbar limits.

    Returns:
        ani (FuncAnimation): The animation object.
    """
    num_steps = grid_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(grid_history[0], cmap=colormap, origin="lower")
    if clim is not None:
        im.set_clim(*clim)
    ax.set_title(title)

    def update(frame):
        im.set_array(grid_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, blit=True, interval=500, repeat=False
    )

    plt.colorbar(im, ax=ax, label=colorbar_label)
    plt.tight_layout()
    if file_name:

        def update_func(_i, _n):
            progress_bar.update(1)

        # update_func = lambda _i, _n: progress_bar.update(1)
        desc = f"Saving video: {title}" if title else "Saving video"
        with tqdm(
            total=num_steps,
            desc=desc,
        ) as progress_bar:
            ani.save(
                get_plot_directory(file_name), dpi=300, progress_callback=update_func
            )
        print("Video saved successfully.\n")
        plt.close()
        plt.close()
    else:
        plt.show()

    return ani


def plot_grid_average_over_time(grid_values, title, xlabel, ylabel, file_name=None):
    """Plot overall agent values over time.

    Args:
        grid_values (np.ndarray): 3D array of shape (num_steps, height,
            width) representing the values to average over time.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        file_name (str, optional): Name of the file to save the plot. If None,
            will display the plot.
    """
    avg_values = np.mean(grid_values, axis=(1, 2))

    plt.plot(avg_values, label="Average Value")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        print(f"Plot saved to {get_plot_directory(file_name)}")
        plt.close()
    else:
        plt.show()


def plot_list_over_time(
    data, title, xlabel, ylabel, file_name=None, legend_labels=None
):
    """Plot a list of data over time.

    Args:
        data (list[np.ndarray]): List of 1D arrays to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        file_name (str, optional): Name of the file to save the plot. If None,
            will display the plot.
        legend_labels (list[str], optional): Labels for each line in the legend.
    """
    plt.plot(data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        print(f"Plot saved to {get_plot_directory(file_name)}")
        plt.close()
    else:
        plt.show()


def plot_mean_and_variability_array(
    data: np.ndarray, title: str, kind: str = "std", file_name: str | None = None
):
    """Plot the mean and variability of a 2D array over time."""
    time = np.arange(data.shape[1])
    mean = np.mean(data, axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(time, mean, label="Mean", color="blue")

    if kind == "std":
        std = np.std(data, axis=0)
        plt.fill_between(
            time, mean - std, mean + std, color="blue", alpha=0.3, label="±1 Std Dev"
        )
    elif kind == "percentile":
        lower = np.percentile(data, 10, axis=0)
        upper = np.percentile(data, 90, axis=0)
        plt.fill_between(
            time, lower, upper, color="blue", alpha=0.3, label="10–90% Range"
        )

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        print(f"Plot saved to {get_plot_directory(file_name)}")
        plt.close()
    else:
        plt.show()


def plot_sobol_indices(Si, time_steps, var_names, output_label, file_name=None):
    """Plot Sobol sensitivity indices for given time steps and variable names."""
    num_vars = len(var_names)

    for t in time_steps:
        fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        fig.suptitle(f"Sobol Sensitivity Analysis for {output_label} at t={t}")

        # First-order indices
        axs[0].bar(
            range(num_vars), Si[t]["S1"], yerr=Si[t].get("S1_conf", None), capsize=5
        )
        axs[0].set_title("First-order Sobol Indices (S1)")
        axs[0].set_xticks(range(num_vars))
        axs[0].set_xticklabels(var_names, rotation=45)
        axs[0].set_ylim(0, 1)

        # Total-order indices
        axs[1].bar(
            range(num_vars), Si[t]["ST"], yerr=Si[t].get("ST_conf", None), capsize=5
        )
        axs[1].set_title("Total-order Sobol Indices (ST)")
        axs[1].set_xticks(range(num_vars))
        axs[1].set_xticklabels(var_names, rotation=45)
        axs[1].set_ylim(0, 1)

        plt.tight_layout()
        file_name = str(t) + "_" + file_name
        if file_name:
            plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
            print(f"Plot saved to {get_plot_directory(file_name)}")
            plt.close()
        else:
            plt.show()


def plot_support_derivative(a: float = 1, b: float = 1, savedir: Path | None = None):
    """Plot the derivative of support for cooperation with respect to environment."""
    savedir = savedir or Path(".")

    fig, axes = plt.subplots(
        ncols=3, figsize=(7, 4), constrained_layout=True, sharey=True
    )

    support = np.array([0, 0.25, 0.5, 0.75, 1.0])
    pessimism = np.array([0.5, 1.0, 2.0])
    n = np.linspace(0, 1, 101)

    def draw(s: float, n_perceived: npt.NDArray[np.float64], ax: Axes):
        logistic = 4 * n_perceived * (1 - n_perceived)
        growth = a * logistic * (1 - s)
        decay = b * (1 - logistic) * s
        ax.plot(n, growth - decay, label=f"$s(t) = {s:.2f}$")

    for ax, pes in zip(axes, pessimism, strict=True):
        for s in support:
            draw(s, n**pes, ax)
        ax.set_xlabel(r"Environment state ($n$)")

    axes[0].set_title(f"Optimistic ($n^* = n^{{{pessimism[0]}}}$)")
    axes[1].set_title(f"Realistic ($n^* = n^{{{pessimism[1]}}}$)")
    axes[2].set_title(f"Pessimistic ($n^* = n^{{{pessimism[2]}}}$)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.01, 1.01)
        ax.axhline(y=0, linestyle="dashed", color="grey", linewidth=1)

    fig.suptitle("Change in support for cooperation, varying environment and pessimism")
    fig.supylabel(r"$\frac{ds}{dn}$", rotation=0)
    fig.savefig(savedir, dpi=300, bbox_inches="tight")
    plt.show()


def plot_phase_portrait(
    c: float,
    recovery: float,
    pollution: float,
    rationality: float = 1,
    gamma_n: float = 0.01,
    gamma_s: float = 0.001,
    equilibria: bool = True,
    dn_dt_nullcline: bool = True,
    dm_dt_nullcline: bool = False,
    critical_points: bool = False,
    ax: Axes | None = None,
    b: float | None = None,
):
    """Draw a phase portrait for a mean-field model.

    Args:
        c: Utility function weight for the 'peer pressure' term.
        recovery: How quickly the environment recovers under positive action.
        pollution: How quickly the environment degrades due to negative action.
        rationality: Controls how rational agents are. Larger is more rational
            (deterministic). 0 is random.
        gamma_n: Scale coefficient for dn/dt, controls the general rate
            of change in the (average) environment.
        gamma_s: Scale coefficient for ds/dt, controls the general rate
            of change in preference for cooperation.
        equilibria: Display equilibria as red circles.
        dn_dt_nullcline: Show dn/dt = 0 as a dashed grey line.
        dm_dt_nullcline: Show dm/dt = 0 as orange (stable) and green (unstable)
            circles.
        critical_points: Show critical points where ds/dt diverges (currently
            unimplemented).
        ax: Optional matplotlib Axes object to draw plot onto. If unspecified, uses
            the current artist.
        b: Optional utility function weight for the 'individual preference' term.
    """
    if b is None:
        b = 1 - c
    ALPHA = 1
    BETA = 1

    # Since P(C) is probabilistic, m is bounded in open interval (-1, 1)
    min_m = mf.fixedpoint_mean_action(0, c, rationality).lower + 1e-3
    max_m = mf.fixedpoint_mean_action(4, c, rationality).upper - 1e-3

    # Axis samples
    ns = np.linspace(0, 1, 100)
    ms = np.linspace(min_m, max_m, 100)

    # Construct derivative functions
    m_prime = mf.f_dm_dt(rationality, b, c, ALPHA, BETA, rate=gamma_s)
    n_prime = mf.f_dn_dt(recovery, pollution, rate=gamma_n)

    # Calculate derivatives for mean action
    N, M = np.meshgrid(ns, ms)
    DM_DT = m_prime(M, N)

    # Calculate derivatives for environment
    pc = (ms + 1) / 2
    N, P = np.meshgrid(ns, pc)
    DN_DT = n_prime(N, P)

    # Plotting
    # =========
    if not ax:
        ax = plt.gca()

    # 1. Plot phase portait
    ax.streamplot(N, M, DN_DT, DM_DT, density=[1.5, 2.5], linewidth=0.5, zorder=1)

    # 2. Draw nullclines
    # m' = 0 when s' = 0
    if dm_dt_nullcline:
        sigma_n = 4 * ns * (1 - ns)
        s = 4 * ALPHA * sigma_n / (BETA * (1 - sigma_n) + ALPHA * sigma_n)
        da_dt_nullcline_ns = []
        da_dt_nullcline_ms = []
        for n, _s in zip(ns, s, strict=True):
            roots = mf.fixedpoint_mean_action(_s, c, rationality)
            for m in roots.stable():
                da_dt_nullcline_ns.append(n)
                da_dt_nullcline_ms.append(m)
        ax.scatter(da_dt_nullcline_ns, da_dt_nullcline_ms, s=10, color="orange")

        da_dt_nullcline_ns = []
        da_dt_nullcline_ms = []
        for n, _s in zip(ns, s, strict=True):
            roots = mf.fixedpoint_mean_action(_s, c, rationality)
            for m in roots.unstable():
                da_dt_nullcline_ns.append(n)
                da_dt_nullcline_ms.append(m)
        ax.scatter(da_dt_nullcline_ns, da_dt_nullcline_ms, s=10, color="green")

    # n' = 0
    if dn_dt_nullcline:
        stationary_ns = (recovery * (1 + ms)) / (
            pollution * (1 - ms) + recovery * (1 + ms)
        )
        ax.plot(
            stationary_ns,
            ms,
            linestyle="dashed",
            color="gray",
            linewidth=0.7,
            label=r"$\frac{dn}{dt} = 0$",
            zorder=0,
        )

    # 3. Plot equilibrium points
    if equilibria:
        eq_n, eq_m = mf.solve_for_equilibria(
            b=b, c=c, rationality=rationality, recovery=recovery, pollution=pollution
        )

        ax.scatter(eq_n, eq_m, color="red", zorder=2)

    # 4. Draw locations where dm/dt diverges (critical points)
    # !! Currently commented because unfinished derivation.
    if critical_points:
        raise NotImplementedError("Plotting ds/dt = 0 is not yet implemented.")
    #    # Check if critical points exist
    #    if c >= 1 / (2 * rationality):
    #        t = np.arccosh(np.sqrt(2 * rationality * c))
    #        mc = [
    #            1 / (2 * c) * ((1 - c) * (2 - s) - t),
    #            1 / (2 * c) * ((1 - c) * (2 - s) + t),
    #        ]
    #        ax.hlines(
    #            y=mc,
    #            xmin=0,
    #            xmax=1,
    #            linewidth=0.7,
    #            linestyle="-.",
    #            color="gray",
    #            label=r"$\frac{dm}{dt} \to \infty$",
    #        )


def get_file_basename(suffix, neighb, severity, radius, b2, env):
    """Generate a standardized file basename based on parameters.

    Args:
        suffix (str): Suffix for the file name.
        neighb (str): Neighborhood type (e.g., "linear", "von_neumann").
        severity (str): Severity level (e.g., "low", "medium", "high").
        radius (int): Radius of the neighborhood.
        b2 (float | None): Second utility function weight, or None if not applicable.
        env (str): Environment update type (e.g., "static", "dynamic").

    Returns:
        str: A standardized file basename.
    """
    return (
        f"{suffix}_neighb-{neighb}_sev-{severity}_rad-{radius}_"
        f"b2-{'1' if b2 is not None else 'None'}_env-{env}"
    )


def save_and_plot_heatmap(
    data,
    title,
    suffix,
    neighb,
    severity,
    radius,
    b2,
    env_update_type,
    savedir,
    rationality_values=None,
    gamma_s_values=None,
):
    """Save and plot a heatmap of the given data.

    Args:
        data (np.ndarray): 2D array of data to plot.
        title (str): Title for the heatmap.
        suffix (str): Suffix for the file name.
        neighb (str): Neighborhood type (e.g., "linear", "von_neumann").
        severity (str): Severity level (e.g., "low", "medium", "high").
        radius (int): Radius of the neighborhood.
        b2 (float | None): Second utility function weight,
            or None if not applicable.
        env_update_type (str): Environment update type (e.g., "static", "dynamic").
        savedir (Path): Directory to save the heatmap and data.
        rationality_values (list[float] | None): List of rationality
            values for x-axis ticks.
        gamma_s_values (list[float] | None): List of support
            update rates for y-axis ticks.
    """
    dir_name = (
        f"neighb-{neighb}_sev-{severity}_rad-{radius}_"
        f"b2-{'1' if b2 is not None else 'None'}_env-{env_update_type}"
    )
    plot_dir = savedir / dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    base = get_file_basename(suffix, neighb, severity, radius, b2, env_update_type)
    data_path = plot_dir / f"data_{base}.npy"
    fig_path = plot_dir / f"heatmap_{base}.png"

    np.save(data_path, data)

    fig_width = 3.5  # inches (single column width)
    fig_height = 2.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    # Check if data is in [0, 1] and set vmin/vmax accordingly
    vmin, vmax = None, None
    if np.nanmin(data) >= 0 and np.nanmax(data) <= 1 and suffix != "peak_freq":
        vmin, vmax = 0, 1

    if rationality_values is not None and gamma_s_values is not None:
        x_tick_indices = np.arange(
            0, len(rationality_values), max(1, len(rationality_values) // 4)
        )
        y_tick_indices = np.arange(
            0, len(gamma_s_values), max(1, len(gamma_s_values) // 4)
        )

        x_tick_labels = [f"{rationality_values[i]:.1f}" for i in x_tick_indices]
        y_tick_labels = [f"{gamma_s_values[i]:.3f}" for i in y_tick_indices]

        sns.heatmap(
            data,
            xticklabels=False,
            yticklabels=False,
            cmap="viridis",
            cbar_kws={"label": title, "shrink": 0.7, "pad": 0.02},
            ax=ax,
            square=False,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels(y_tick_labels)
    else:
        sns.heatmap(
            data,
            xticklabels=4,
            yticklabels=4,
            cmap="viridis",
            cbar_kws={"label": title, "shrink": 0.7, "pad": 0.02},
            ax=ax,
            square=False,
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_xlabel("Rationality (λ)", fontsize=9)
    ax.set_ylabel("Support Update Rate (γₛ)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

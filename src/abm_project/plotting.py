"""Plotting functions for agent-based model visualizations."""

import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from tqdm import tqdm


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

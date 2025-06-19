"""Plotting functions for agent-based model visualizations."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


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
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
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
        ani.save(get_plot_directory(file_name))
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
        plt.close()
    else:
        plt.show()

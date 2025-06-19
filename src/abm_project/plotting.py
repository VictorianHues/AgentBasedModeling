"""Plotting functions for agent-based model visualizations."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def get_plot_directory(file_name):
    """Get the directory for saving plots."""
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_dir = os.path.join(plot_dir, file_name) if file_name else plot_dir
    return plot_dir


def plot_current_agent_env_status(model, file_name=None):
    """Plot the environment status of agents."""
    env_status_grid = np.zeros((model.width, model.height))
    for x in range(model.width):
        for y in range(model.height):
            env_status_grid[x, y] = model.agents[x, y].get_recent_env_status()

    plt.imshow(env_status_grid, cmap="RdYlGn", origin="lower")
    plt.colorbar(label="Environment Status")
    plt.title("Agent Environment Status")
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_current_agent_actions(model, file_name=None):
    """Plot the actions of agents."""
    action_grid = np.zeros((model.width, model.height))
    for x in range(model.width):
        for y in range(model.height):
            action_grid[x, y] = model.agents[x, y].get_recent_action()

    plt.imshow(action_grid, cmap=ListedColormap(["red", "green"]), origin="lower")
    plt.colorbar(label="Agent Action (-1 or 1)")
    plt.title("Agent Actions")
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def animate_agent_actions(model, file_name=None):
    """Animate the environment status of agents over a number of steps."""
    agent_action_history = np.array(model.agent_action_history)
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
        fig, update, frames=num_steps, blit=True, interval=500, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Agent Action (-1 or 1)")
    plt.tight_layout()
    if file_name:
        ani.save(get_plot_directory(file_name))
        plt.close()
    else:
        plt.show()

    return ani


def animate_agent_env_status(model, file_name=None):
    """Animate the environment status of agents over a number of steps."""
    env_status_history = np.array(model.agent_env_status_history)
    num_steps = env_status_history.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(env_status_history[0], cmap="RdYlGn", origin="lower")
    ax.set_title("Agent Environment Status Over Time")

    def update(frame):
        im.set_array(env_status_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, blit=True, interval=500, repeat=False
    )

    plt.colorbar(im, ax=ax, label="Environment Status")
    plt.tight_layout()
    if file_name:
        ani.save(get_plot_directory(file_name))
        plt.close()
    else:
        plt.show()

    return ani


def plot_overall_agent_actions_over_time(model, file_name=None):
    """Plot the overall agent actions over time."""
    action_history = np.array(model.agent_action_history)
    avg_actions = np.mean(action_history, axis=(1, 2))

    plt.plot(avg_actions, label="Average Agent Action")
    plt.xlabel("Time Step")
    plt.ylabel("Average Action (-1 or 1)")
    plt.title("Overall Agent Actions Over Time")
    plt.axhline(0, color="gray", linestyle="--")
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_overall_agent_env_status_over_time(model, file_name=None):
    """Plot the overall agent environment status over time."""
    env_status_history = np.array(model.agent_env_status_history)
    avg_env_status = np.mean(env_status_history, axis=(1, 2))

    plt.plot(avg_env_status, label="Average Environment Status")
    plt.xlabel("Time Step")
    plt.ylabel("Average Environment Status")
    plt.title("Overall Agent Environment Status Over Time")
    plt.axhline(0, color="gray", linestyle="--")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if file_name:
        plt.savefig(get_plot_directory(file_name), bbox_inches="tight")
        plt.close()
    else:
        plt.show()

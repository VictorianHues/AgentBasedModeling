"""Plotting functions for agent-based model visualizations."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_current_agent_env_status(model):
    """Plot the environment status of agents."""
    env_status_grid = np.zeros((model.width, model.height))
    for x in range(model.width):
        for y in range(model.height):
            env_status_grid[x, y] = model.agents[x, y].env_status

    plt.imshow(env_status_grid, cmap="RdYlGn", origin="lower")
    plt.colorbar(label="Environment Status")
    plt.title("Agent Environment Status")
    plt.show()


def plot_current_agent_actions(model):
    """Plot the actions of agents."""
    action_grid = np.zeros((model.width, model.height))
    for x in range(model.width):
        for y in range(model.height):
            action_grid[x, y] = model.agents[x, y].action

    plt.imshow(action_grid, cmap=ListedColormap(["red", "green"]), origin="lower")
    plt.colorbar(label="Agent Action (-1 or 1)")
    plt.title("Agent Actions")
    plt.show()


def animate_agent_actions(model):
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
    plt.show()

    return ani


def animate_agent_env_status(model):
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
    plt.show()

    return ani

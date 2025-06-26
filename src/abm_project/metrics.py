"""Various model metrics."""

import networkx as nx
import numpy as np

from abm_project.oop_model import BaseModel


def extract_actions(model):
    """Extract OOP model actions matrix as 2D (time, action) array."""
    a = np.asarray(model.agent_action_history)
    a = np.reshape(a, (a.shape[0], -1)).T
    return a


def extract_environment(model):
    """Extract OOP model env matrix as 2D (time, environment) array."""
    n = np.asarray(model.agent_env_status_history)
    n = np.reshape(n, (n.shape[0], -1)).T
    return n


def extract_severity(model):
    """Extract each agents' severity coefficient into a 1D array."""
    b_severity = np.vectorize(lambda agent: agent.env_perception_coeff[-1])(
        model.agents
    ).flatten()
    return b_severity


def extract_peer_pressure(model):
    """Extract each agents' peer pressure coefficient into a 1D array."""
    b_peer_pressure = np.vectorize(lambda agent: agent.peer_pressure_coeff[-1])(
        model.agents
    ).flatten()
    return b_peer_pressure


def extract_adjacency(model):
    """Create adjacency matrix for agents in an OOP model."""
    # Create adjacency matrix
    network = nx.grid_2d_graph(model.width, model.height, periodic=True)
    for i in range(model.width):
        for j in range(model.height):
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                ni %= model.width
                nj %= model.height
                if 0 <= ni < model.width and 0 <= nj < model.height:
                    network.add_edge((i, j), (ni, nj))

    adj = nx.adjacency_matrix(network)
    adj = adj / adj.sum(axis=1)[:, None]

    return adj


def expected_action_volatility(model):
    """Compute the expected volatility in agent actions across simulation.

    For each agent, measures the difference in their actions between each
    pair of consecutive timesteps. The volatility per-agent is calculated
    as the variance over these differences.

    The expected action volatility is then the average of these variances
    across agents.
    """
    a = extract_actions(model)
    a = ((a + 1) / 2).astype(int)
    da_dt = np.diff(a, axis=1)
    action_variance_over_time = np.var(da_dt, axis=1)
    expected_action_variance = np.mean(action_variance_over_time)
    return expected_action_variance


def expected_environment_volatility(model):
    """Compute the expected volatility in agent environment across simulation.

    For each agent, measures the difference in their environment between each
    pair of consecutive timesteps. The volatility per-agent is calculated
    as the variance over these differences.

    The expected environment volatility is then the average of these variances
    across agents.
    """
    n = extract_environment(model)
    dn_dt = np.diff(n, axis=1)
    env_variance_over_time = np.var(dn_dt, axis=1)
    expected_env_variance = np.mean(env_variance_over_time)
    return expected_env_variance


def action_variance_over_time(model):
    """Variance in mean action across simulation."""
    a = extract_actions(model)
    a = ((a + 1) / 2).astype(int)
    mean = a.mean(axis=0)
    var = np.var(a, axis=0)
    return mean, var


def environment_variance_over_time(model):
    """Variance in mean environment across simulation."""
    n = extract_environment(model)
    mean = n.mean(axis=0)
    var = np.var(n, axis=0)
    return mean, var


def cumulative_environmental_harm(model):
    """Cumulative sum of average environmental state."""
    n = (extract_environment(model) * 2 - 1).mean(axis=0)
    return np.cumulative_sum(n)


def probability_repeated_action(model: BaseModel, k: int, warmup: int = 10):
    """For each agent, the decay in probability of repeated cooperation after first."""
    n = extract_environment(model)[:, warmup:]
    a = extract_actions(model)[:, warmup:]
    b_severity = extract_severity(model)
    b_peer_pressure = extract_peer_pressure(model)
    adj = extract_adjacency(model)

    # Compute mean neighbor action at each iteration
    abar = adj @ a

    # Calculate probability of cooperation at each timestep
    v_action = representative_utility(1, abar, n, b_peer_pressure, b_severity)
    v_inaction = representative_utility(-1, abar, n, b_peer_pressure, b_severity)
    p_cooperate = np.exp(v_action) / (np.exp(v_action) + np.exp(v_inaction))

    # For each agent, identify the first time they cooperate
    first_cooperation = np.argmax(a == 1, axis=1)

    # Extract the k probabilities following this
    extracted = np.zeros((model.num_agents, k + 1), dtype=np.float64)
    for agent, t in enumerate(first_cooperation):
        extracted[agent] = p_cooperate[agent, t : t + k + 1]

    return extracted, b_peer_pressure


def pluralistic_ignorance(model: BaseModel):
    """Difference between probability of cooperation with/without social norms."""
    n = extract_environment(model)
    a = extract_actions(model)
    b_severity = extract_severity(model)
    b_peer_pressure = extract_peer_pressure(model)
    adj = extract_adjacency(model)

    # Compute mean neighbor action at each iteration
    abar = adj @ a

    # Calculate actual action probability
    v_action = representative_utility(1, abar, n, b_peer_pressure, b_severity)
    v_inaction = representative_utility(-1, abar, n, b_peer_pressure, b_severity)
    p_actual = np.exp(v_action) / (np.exp(v_action) + np.exp(v_inaction))

    # Calculate probability wihout social norms
    ones = np.full_like(b_severity, fill_value=1)
    zeros = np.full_like(b_severity, fill_value=0)
    v_action = representative_utility(1, abar, n, zeros, ones)
    v_inaction = representative_utility(-1, abar, n, zeros, ones)
    p_ideal = np.exp(v_action) / (np.exp(v_action) + np.exp(v_inaction))

    return p_actual, p_ideal, n.mean(axis=0), a.mean(axis=0)


def ab_gap(model: BaseModel):
    """Difference between probability of cooperation and observed action.

    Per-agent, per-timestep.
    """
    n = extract_environment(model)
    a = extract_actions(model)
    b_severity = extract_severity(model)
    b_peer_pressure = extract_peer_pressure(model)
    adj = extract_adjacency(model)

    # Compute mean neighbor action at each iteration
    abar = adj @ a

    # Calculate representative utility
    v_action = representative_utility(1, abar, n, b_peer_pressure, b_severity)
    v_inaction = representative_utility(-1, abar, n, b_peer_pressure, b_severity)
    p = np.exp(v_action) / (np.exp(v_action) + np.exp(v_inaction))

    gap = p - ((a + 1) / 2).astype(int)
    return gap, b_peer_pressure, b_severity


def representative_utility(action, abar, n, peer_pressure, severity):
    """Compute retrospective representative utility."""
    deviation_cost = peer_pressure[:, None] * (action - abar) ** 2
    perceived_severity = severity[:, None] * (2 * n - 1) * -1.0
    u_action = action * perceived_severity
    v_action = u_action - deviation_cost
    return v_action

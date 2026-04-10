import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def save_grid_animation(
    grid_history: np.ndarray,
    output_path: Path,
    title: str,
    cmap,
    colorbar_label: str,
    fps: int,
    vmin: float | None = None,
    vmax: float | None = None,
):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    im = ax.imshow(
        grid_history[0],
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)

    def update(frame: int):
        im.set_array(grid_history[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=grid_history.shape[0],
        blit=True,
        interval=max(1, int(1000 / fps)),
        repeat=False,
    )

    plt.colorbar(im, ax=ax, label=colorbar_label)
    ani.save(output_path, dpi=150)
    plt.close(fig)


def main(
    lmbda: float,
    gamma_s: float,
    steps: int,
    width: int,
    height: int,
    memory_count: int,
    neighborhood_prediction: str | None,
    const_peer_pressure: bool,
    seed: int,
    fps: int,
    output_prefix: str,
):
    output_dir = Path("results/animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_agents = width * height
    b_2 = np.full(num_agents, 1.0) if const_peer_pressure else None

    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=np.random.default_rng(seed),
        env_update_fn=piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01),
        rationality=lmbda,
        neighb_prediction_option=neighborhood_prediction,
        severity_benefit_option="adaptive",
        b_1=np.full(num_agents, 1.0),
        b_2=b_2,
        max_storage=steps,
        gamma_s=gamma_s,
    )
    model.run(steps)

    env_history = model.environment[: model.time + 1].reshape((-1, height, width))
    action_history = model.action[: model.time + 1].reshape((-1, height, width))

    predict_label = (
        neighborhood_prediction if neighborhood_prediction else "nonpredictive"
    )
    peer_label = "peerconst" if const_peer_pressure else "peerrandomised"
    suffix = (
        f"{output_prefix}_lambda{lmbda:.2f}_gamma{gamma_s:.4f}"
        f"_{predict_label}_{peer_label}"
    )

    env_output = output_dir / f"{suffix}_environment.mp4"
    action_output = output_dir / f"{suffix}_actions.mp4"

    save_grid_animation(
        grid_history=env_history,
        output_path=env_output,
        title=f"Environment (lambda={lmbda:.2f}, gamma_s={gamma_s:.4f})",
        cmap="RdYlGn",
        colorbar_label="Environment status",
        fps=fps,
        vmin=0,
        vmax=1,
    )

    save_grid_animation(
        grid_history=action_history,
        output_path=action_output,
        title=f"Actions (lambda={lmbda:.2f}, gamma_s={gamma_s:.4f})",
        cmap=ListedColormap(["red", "green"]),
        colorbar_label="Action (-1/1)",
        fps=fps,
        vmin=-1,
        vmax=1,
    )

    print(f"Saved environment animation: {env_output}")
    print(f"Saved action animation: {action_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create environment and action animations at a selected "
            "tipping-point parameter set."
        )
    )
    parser.add_argument("--lambda-value", type=float, default=4.0)
    parser.add_argument("--gamma-s", type=float, default=0.0091)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--memory-count", type=int, default=10)
    parser.add_argument("--neighborhood-prediction", default=None)
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output-prefix", default="tipping_point")
    args = parser.parse_args()

    main(
        lmbda=args.lambda_value,
        gamma_s=args.gamma_s,
        steps=args.steps,
        width=args.width,
        height=args.height,
        memory_count=args.memory_count,
        neighborhood_prediction=args.neighborhood_prediction,
        const_peer_pressure=args.const_peer_pressure,
        seed=args.seed,
        fps=args.fps,
        output_prefix=args.output_prefix,
    )

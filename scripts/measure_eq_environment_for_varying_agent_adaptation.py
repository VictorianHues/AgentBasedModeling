import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def run_model(args):
    (
        r,
        i,
        lmbda,
        gamma_s,
        width,
        height,
        steps,
    ) = args

    env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    rng = np.random.default_rng(r)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=lmbda,
        neighb_prediction_option=None,
        max_storage=steps,
        gamma_s=gamma_s,
    )
    model.run(steps)

    final_env = model.environment[model.time]
    mean_env = final_env.mean()

    return (r, i, mean_env)


def main(
    savedir: Path,
    quality_label: str,
    rationality: float,
    repeats: int,
    min_gamma_s: float,
    max_gamma_s: float,
    gamma_steps: int,
    width: int,
    height: int,
    steps: int,
):
    gamma_s_values = np.linspace(min_gamma_s, max_gamma_s, gamma_steps)
    results_env = np.empty((repeats, gamma_steps))

    tasks = [
        (
            r,
            i,
            rationality,
            gamma_s,
            width,
            height,
            steps,
        )
        for r in range(repeats)
        for i, gamma_s in enumerate(gamma_s_values)
    ]

    n_tasks = repeats * gamma_steps
    with ProcessPoolExecutor() as executor:
        for r, i, mean_env in tqdm(executor.map(run_model, tasks), total=n_tasks):
            results_env[r, i] = mean_env

    np.savez(
        savedir / f"eq_env_vs_gamma_s_{quality_label}_quality.npz",
        mean_environment=results_env,
        gamma_s=gamma_s_values,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    RATIONAITY = 4
    MIN_GAMMA_S = 0.001
    MAX_GAMMA_S = 0.02
    WIDTH = 30
    HEIGHT = 30
    STEPS = 1000

    QUICK_REPEATS = 1
    QUICK_GAMMA_STEPS = 5

    FULL_REPEATS = 50
    FULL_GAMMA_STEPS = 25

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
        gamma_steps = QUICK_GAMMA_STEPS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS
        gamma_steps = FULL_GAMMA_STEPS

    main(
        savedir=DATA_DIR,
        quality_label=quality_label,
        repeats=repeats,
        rationality=RATIONAITY,
        min_gamma_s=MIN_GAMMA_S,
        max_gamma_s=MAX_GAMMA_S,
        gamma_steps=gamma_steps,
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
    )

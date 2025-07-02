import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def run_model_with_timeseries(args):
    (r, i, λ, gamma_s, width, height, steps) = args
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = np.random.default_rng(r)
    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=λ,
        neighb_prediction_option=None,
        max_storage=steps,
        gamma_s=gamma_s,
    )
    model.run(steps)
    mean_environment = model.environment[: model.time + 1].mean(axis=1)
    return (r, i, mean_environment)


def main(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: float,
    width: int,
    height: int,
    steps: int,
):
    rationality = np.array([1.0, 3.0, 4.7, 5.5, 6.5])
    mean_environment = np.empty(
        (repeats, len(rationality), steps + 1), dtype=np.float64
    )

    with ProcessPoolExecutor() as executor:
        futures = []
        for r in range(repeats):
            for λi, λ in enumerate(rationality):
                future = executor.submit(
                    run_model_with_timeseries, (r, λi, λ, gamma_s, width, height, steps)
                )
                futures.append(future)

        desc = "Collecting mean environment time series data for varying rationality"
        n_tasks = repeats * len(rationality)
        for future in tqdm(as_completed(futures), desc=desc, total=n_tasks):
            (repeat, λi, val) = future.result()
            mean_environment[repeat, λi] = val

    np.savez(
        data_dir
        / f"time_series_mean_env_for_varying_rationality_{quality_label}_quality.npz",
        mean_environment=mean_environment,
        rationality=rationality,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    GAMMA_S = 0.001
    WIDTH = 30
    HEIGHT = 30
    N_STEPS = 1000

    QUICK_REPEATS = 1
    FULL_REPEATS = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS

    main(
        data_dir=DATA_DIR,
        quality_label=quality_label,
        repeats=repeats,
        gamma_s=GAMMA_S,
        width=WIDTH,
        height=HEIGHT,
        steps=N_STEPS,
    )

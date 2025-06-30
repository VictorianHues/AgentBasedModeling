import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.batch_run_tools import (
    analyze_environment_clusters_periodic,
    get_dominant_frequency_and_power,
)
from abm_project.utils import linear_update, piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def run_model(args):
    (
        r,
        i,
        lmbda,
        gamma_s,
        neighb_prediction_option,
        severity_benefit_option,
        radius_option,
        b_2,
        env_update_fn_type,
        memory_count,
        width,
        height,
        steps,
    ) = args

    if env_update_fn_type == "linear":
        env_update_fn = linear_update(0.01)
    elif env_update_fn_type == "piecewise":
        env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    else:
        raise ValueError(f"Unknown env_update_fn_type: {env_update_fn_type}")

    rng = np.random.default_rng(r)
    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=lmbda,
        simmer_time=1,
        neighb_prediction_option=neighb_prediction_option,
        severity_benefit_option=severity_benefit_option,
        radius_option=radius_option,
        max_storage=steps,
        b_1=np.full(width * height, 1.0),
        b_2=b_2 if b_2 is not None else None,
        gamma_s=gamma_s,
    )
    model.run(steps)

    final_env = model.environment[model.time]
    mean_env = final_env.mean()

    num_clusters, cluster_sizes, _ = analyze_environment_clusters_periodic(
        final_env, width, height, threshold=0.5
    )
    mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
    max_cluster_size = np.max(cluster_sizes) if cluster_sizes else 0

    env_timeseries = model.environment[: model.time].mean(axis=1)
    peak_freq, power = get_dominant_frequency_and_power(env_timeseries)

    return (
        r,
        i,
        num_clusters,
        mean_cluster_size,
        max_cluster_size,
        mean_env,
        peak_freq,
        power,
    )


def main(
    savedir: Path,
    gamma_s: float = 0.01,
    memory_count_single: int = 10,
    neighb_prediction_option=None,
    severity_benefit_option=None,
    radius_option="single",
    b_2=None,
    env_update_fn_type="linear",
    repeats: int = 50,
    min_rationality: float = 0.0,
    max_rationality: float = 5.0,
    rationality_steps: int = 25,
    quality_label: str = "low",
    width: int = 30,
    height: int = 30,
    steps: int = 1000,
):
    rationality = np.linspace(min_rationality, max_rationality, rationality_steps)

    results = np.empty((repeats, len(rationality)), dtype=np.float64)

    tasks = [
        (
            r,
            i,
            lmbda,
            gamma_s,
            neighb_prediction_option,
            severity_benefit_option,
            radius_option,
            b_2,
            env_update_fn_type,
            memory_count_single,
            width,
            height,
            steps,
        )
        for r in range(repeats)
        for i, lmbda in enumerate(rationality)
    ]

    n_tasks = repeats * len(rationality)
    with ProcessPoolExecutor() as executor:
        for r, i, _, _, _, mean_env, _, _ in tqdm(
            executor.map(run_model, tasks), total=n_tasks
        ):
            results[r, i] = mean_env

    np.savez(
        savedir / f"eq_env_vs_rationality_{quality_label}_quality.npz",
        mean_environment=results,
        rationality=rationality,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    GAMMA_S = 1.5
    MEMORY_SIZE = 10
    NEIGHBORHOOD_PREDICTION = None
    SEVERITY = None
    RADIUS = "single"
    B2 = np.full(900, 1.0)
    ENV_UPDATE_FN = "piecewise"
    REPEATS = 10
    MIN_RATIONALITY = 0.0
    MAX_RATIONALITY = 8.0
    RATIONALITY_STEPS = 25
    WIDTH = 30
    HEIGHT = 30
    STEPS = 1000

    QUICK_REPEATS = 1
    QUICK_RATIONALITY_STEPS = 5

    FULL_REPEATS = 10
    FULL_RATIONALITY_STEPS = 25

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
        rationality_steps = QUICK_RATIONALITY_STEPS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS
        rationality_steps = FULL_RATIONALITY_STEPS

    main(
        savedir=DATA_DIR,
        gamma_s=GAMMA_S,
        memory_count_single=MEMORY_SIZE,
        neighb_prediction_option=NEIGHBORHOOD_PREDICTION,
        severity_benefit_option=SEVERITY,
        radius_option=RADIUS,
        b_2=B2,
        env_update_fn_type=ENV_UPDATE_FN,
        repeats=repeats,
        min_rationality=MIN_RATIONALITY,
        max_rationality=MAX_RATIONALITY,
        rationality_steps=rationality_steps,
        quality_label=quality_label,
        width=30,
        height=30,
        steps=1000,
    )

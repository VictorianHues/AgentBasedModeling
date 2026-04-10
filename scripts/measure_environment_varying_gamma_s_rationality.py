import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from abm_project.batch_run_tools import (
    compute_autocorrelation,
    compute_time_averaged_variance,
    estimate_autocorrelation_time,
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def get_parallel_workers(default_workers: int = 12) -> int:
    """Return a conservative process count; override with ABM_MAX_WORKERS."""
    cpu_count = os.cpu_count() or 1
    env_value = os.getenv("ABM_MAX_WORKERS")
    if env_value is not None:
        try:
            return max(1, min(int(env_value), cpu_count))
        except ValueError:
            print(
                "Invalid ABM_MAX_WORKERS="
                f"'{env_value}'. Falling back to "
                f"{default_workers}."
            )
    return max(1, min(default_workers, cpu_count))


def get_chunksize(n_tasks: int, workers: int, default: int = 1) -> int:
    """Get executor map chunksize; override with ABM_CHUNKSIZE."""
    env_value = os.getenv("ABM_CHUNKSIZE")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            print(f"Invalid ABM_CHUNKSIZE='{env_value}'. Falling back to auto.")
    if n_tasks <= 0:
        return 1
    return max(1, min(default, n_tasks // max(1, workers * 8)))


def measure_equilibrium_environment(args):
    (
        base_seed,
        r,
        γi,
        λi,
        γ,
        λ,
        neighb_predict_option,
        dynamic_action_preference,
        peer_pressure_coeff,
        memory_count,
        width,
        height,
        steps,
        variance_last_k,
        acf_max_lag,
    ) = args
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    seed_seq = np.random.SeedSequence([base_seed, r, γi, λi])
    rng = np.random.default_rng(seed_seq)
    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=λ,
        neighb_prediction_option=neighb_predict_option,
        severity_benefit_option=dynamic_action_preference,
        b_1=np.full(width * height, 1.0),
        b_2=peer_pressure_coeff,
        max_storage=steps,
        gamma_s=γ,
    )
    model.run(steps)
    final_env = model.environment[model.time]
    mean_env = final_env.mean()
    env_history = model.environment[: model.time + 1]
    env_timeseries = model.environment[: model.time].mean(axis=1)
    spatial_var = compute_time_averaged_variance(env_history, last_k=variance_last_k)
    acf = compute_autocorrelation(env_timeseries, max_lag=acf_max_lag)
    ac_time = estimate_autocorrelation_time(acf)
    return (r, γi, λi, mean_env, spatial_var, ac_time, acf)


def main(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    neighborhood_prediction_option: str,
    dynamic_action_preference: str | None,
    peer_pressure_coeff: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
    memory_count: int,
    variance_last_k: int,
    acf_max_lag: int,
    base_seed: int,
):
    mean_environment = np.empty(
        (repeats, len(gamma_s), len(rationality)), dtype=np.float64
    )
    spatial_variance = np.empty(
        (repeats, len(gamma_s), len(rationality)), dtype=np.float64
    )
    autocorrelation_time = np.empty(
        (repeats, len(gamma_s), len(rationality)), dtype=np.float64
    )

    lag_max = max(0, min(int(acf_max_lag), steps - 1))
    acf_sum = np.zeros((len(gamma_s), len(rationality), lag_max + 1), dtype=np.float64)
    acf_count = np.zeros((len(gamma_s), len(rationality)), dtype=np.int64)

    desc = (
        "Collecting equilibrium mean environment, spatial variance, and temporal"
        " autocorrelation for varying rationality and agent adaptation"
    )
    n_tasks = repeats * len(rationality) * len(gamma_s)
    workers = get_parallel_workers()
    chunksize = get_chunksize(n_tasks=n_tasks, workers=workers)

    tasks = (
        (
            base_seed,
            r,
            γi,
            λi,
            γ,
            λ,
            neighborhood_prediction_option,
            dynamic_action_preference,
            peer_pressure_coeff,
            memory_count,
            width,
            height,
            steps,
            variance_last_k,
            lag_max,
        )
        for r in range(repeats)
        for γi, γ in enumerate(gamma_s)
        for λi, λ in enumerate(rationality)
    )

    print(f"Running {n_tasks} simulations with {workers} worker process(es).")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for repeat, γi, λi, mean_val, var_val, ac_time, acf in tqdm(
            executor.map(measure_equilibrium_environment, tasks, chunksize=chunksize),
            desc=desc,
            total=n_tasks,
        ):
            mean_environment[repeat, γi, λi] = mean_val
            spatial_variance[repeat, γi, λi] = var_val
            autocorrelation_time[repeat, γi, λi] = ac_time
            acf_sum[γi, λi] += acf
            acf_count[γi, λi] += 1

    mean_autocorrelation = np.full_like(acf_sum, np.nan)
    valid = acf_count > 0
    mean_autocorrelation[valid] = acf_sum[valid] / acf_count[valid, None]

    peer_pressure_label = (
        "peerconst" if peer_pressure_coeff is not None else "peerrandomised"
    )
    predict_label = "predictive" if neighborhood_prediction_option else "nonpredictive"
    adaptive_label = dynamic_action_preference or "nonadaptive"
    if adaptive_label == "adaptive":
        # Preserve existing output filename for backward compatibility.
        filename = (
            f"equilibrium_env_rationality_vs_gamma_s"
            f"_{predict_label}_{peer_pressure_label}"
            f"_{quality_label}_quality.npz"
        )
    else:
        filename = (
            f"equilibrium_env_rationality_vs_gamma_s"
            f"_{adaptive_label}_{predict_label}_{peer_pressure_label}"
            f"_{quality_label}_quality.npz"
        )
    np.savez(
        data_dir / filename,
        mean_environment=mean_environment,
        spatial_variance=spatial_variance,
        autocorrelation_time=autocorrelation_time,
        mean_autocorrelation=mean_autocorrelation,
        acf_lags=np.arange(lag_max + 1, dtype=np.int64),
        gamma_s=gamma_s,
        rationality=rationality,
        variance_last_k=variance_last_k,
        acf_max_lag=lag_max,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    MIN_GAMMA_S = 0.001
    MAX_GAMMA_S = 0.02
    MIN_RATIONALITY = 0.0
    MAX_RATIONALITY = 6.0

    WIDTH = 30
    HEIGHT = 30
    N_STEPS = 1000
    MEMORY_COUNT = 10
    CONST_PEER_PRESSURE_WEIGHT = np.full(WIDTH * HEIGHT, 1.0)

    QUICK_REPEATS = 3
    QUICK_GAMMA_S_STEPS = 25
    QUICK_RATIONALITY_STEPS = 25

    FULL_REPEATS = 20
    FULL_GAMMA_S_STEPS = 25
    FULL_RATIONALITY_STEPS = 25

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument(
        "--dynamic-action-preference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use adaptive action preference dynamics. Disabled by default; pass "
            "--dynamic-action-preference for adaptive runs."
        ),
    )
    parser.add_argument("--neighborhood-prediction", default=None)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help=(
            "Base seed used to derive unique RNG streams for each "
            "(repeat, gamma_s, rationality) task."
        ),
    )
    parser.add_argument("--variance-last-k", type=int, default=1)
    parser.add_argument("--acf-max-lag", type=int, default=200)
    parser.add_argument("--gamma-s-steps", type=int)
    parser.add_argument("--rationality-steps", type=int)
    args = parser.parse_args()

    if args.base_seed < 0:
        parser.error("--base-seed must be >= 0")

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
        gamma_s_steps = QUICK_GAMMA_S_STEPS
        rationality_steps = QUICK_RATIONALITY_STEPS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS
        gamma_s_steps = FULL_GAMMA_S_STEPS
        rationality_steps = FULL_RATIONALITY_STEPS

    if args.gamma_s_steps is not None:
        if args.gamma_s_steps < 2:
            parser.error("--gamma-s-steps must be >= 2")
        gamma_s_steps = args.gamma_s_steps

    if args.rationality_steps is not None:
        if args.rationality_steps < 2:
            parser.error("--rationality-steps must be >= 2")
        rationality_steps = args.rationality_steps

    peer_pressure_coeff = None
    if args.const_peer_pressure:
        peer_pressure_coeff = CONST_PEER_PRESSURE_WEIGHT

    dynamic_action_preference = None
    if args.dynamic_action_preference:
        dynamic_action_preference = "adaptive"

    gamma_s = np.linspace(MIN_GAMMA_S, MAX_GAMMA_S, gamma_s_steps)
    rationality = np.linspace(MIN_RATIONALITY, MAX_RATIONALITY, rationality_steps)

    main(
        data_dir=DATA_DIR,
        quality_label=quality_label,
        repeats=repeats,
        gamma_s=gamma_s,
        rationality=rationality,
        neighborhood_prediction_option=args.neighborhood_prediction,
        dynamic_action_preference=dynamic_action_preference,
        peer_pressure_coeff=peer_pressure_coeff,
        width=WIDTH,
        height=HEIGHT,
        steps=N_STEPS,
        memory_count=MEMORY_COUNT,
        variance_last_k=args.variance_last_k,
        acf_max_lag=args.acf_max_lag,
        base_seed=args.base_seed,
    )

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
    get_dominant_frequency_and_power,
)
from abm_project.utils import linear_update, piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel

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

HEATMAP_QUICK_REPEATS = 3
HEATMAP_QUICK_GAMMA_S_STEPS = 25
HEATMAP_QUICK_RATIONALITY_STEPS = 25

HEATMAP_FULL_REPEATS = 20
HEATMAP_FULL_GAMMA_S_STEPS = 25
HEATMAP_FULL_RATIONALITY_STEPS = 25

EQ_ENV_VS_RATIONALITY_GAMMA_S = 1.5
EQ_ENV_VS_RATIONALITY_MEMORY_SIZE = 10
EQ_ENV_VS_RATIONALITY_NEIGHBORHOOD_PREDICTION = None
EQ_ENV_VS_RATIONALITY_SEVERITY = None
EQ_ENV_VS_RATIONALITY_RADIUS = "single"
EQ_ENV_VS_RATIONALITY_B2 = np.full(WIDTH * HEIGHT, 1.0)
EQ_ENV_VS_RATIONALITY_ENV_UPDATE_FN = "piecewise"
EQ_ENV_VS_RATIONALITY_MIN = 0.0
EQ_ENV_VS_RATIONALITY_MAX = 8.0
EQ_ENV_VS_RATIONALITY_QUICK_REPEATS = 5
EQ_ENV_VS_RATIONALITY_QUICK_STEPS = 50
EQ_ENV_VS_RATIONALITY_FULL_REPEATS = 1000
EQ_ENV_VS_RATIONALITY_FULL_STEPS = 50

EQ_ENV_VS_GAMMA_RATIONALITY = 4
EQ_ENV_VS_GAMMA_QUICK_REPEATS = 5
EQ_ENV_VS_GAMMA_QUICK_STEPS = 50
EQ_ENV_VS_GAMMA_FULL_REPEATS = 100
EQ_ENV_VS_GAMMA_FULL_STEPS = 50
EQ_ENV_VS_GAMMA_STEPS = 2000

FOURIER_VS_GAMMA_RATIONALITY = 4
FOURIER_VS_GAMMA_QUICK_REPEATS = 5
FOURIER_VS_GAMMA_QUICK_STEPS = 50
FOURIER_VS_GAMMA_FULL_REPEATS = 100
FOURIER_VS_GAMMA_FULL_STEPS = 50
FOURIER_VS_GAMMA_STEPS = 1000

TIME_SERIES_GAMMA_S = 0.015
TIME_SERIES_STEPS = 2000
TIME_SERIES_QUICK_REPEATS = 1
TIME_SERIES_FULL_REPEATS = 50
TIME_SERIES_RATIONALITY_VALUES = np.array([1.0, 3.0, 4.3, 5.5, 6.5], dtype=np.float64)


def get_parallel_workers(default_workers: int = 12) -> int:
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
    env_value = os.getenv("ABM_CHUNKSIZE")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            print(f"Invalid ABM_CHUNKSIZE='{env_value}'. Falling back to auto.")
    if n_tasks <= 0:
        return 1
    return max(1, min(default, n_tasks // max(1, workers * 8)))


def _measure_environment_heatmap_task(args: tuple):
    (
        base_seed,
        repeat_idx,
        gamma_idx,
        rationality_idx,
        gamma_value,
        rationality_value,
        neighborhood_prediction_option,
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
    seed_seq = np.random.SeedSequence(
        [base_seed, repeat_idx, gamma_idx, rationality_idx]
    )
    rng = np.random.default_rng(seed_seq)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality_value,
        neighb_prediction_option=neighborhood_prediction_option,
        severity_benefit_option=dynamic_action_preference,
        b_1=np.full(width * height, 1.0),
        b_2=peer_pressure_coeff,
        max_storage=steps,
        gamma_s=gamma_value,
    )

    model.run(steps)
    final_env = model.environment[model.time]
    mean_env = final_env.mean()

    env_history = model.environment[: model.time + 1]
    env_timeseries = model.environment[: model.time].mean(axis=1)
    spatial_var = compute_time_averaged_variance(env_history, last_k=variance_last_k)
    acf = compute_autocorrelation(env_timeseries, max_lag=acf_max_lag)
    ac_time = estimate_autocorrelation_time(acf)

    return (repeat_idx, gamma_idx, rationality_idx, mean_env, spatial_var, ac_time, acf)


def run_measure_environment_heatmap(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    neighborhood_prediction_option: str | None,
    dynamic_action_preference: str | None,
    peer_pressure_coeff: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
    memory_count: int,
    variance_last_k: int,
    acf_max_lag: int,
    base_seed: int,
) -> None:
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

    n_tasks = repeats * len(gamma_s) * len(rationality)
    workers = get_parallel_workers()
    chunksize = get_chunksize(n_tasks=n_tasks, workers=workers)

    tasks = (
        (
            base_seed,
            repeat_idx,
            gamma_idx,
            rationality_idx,
            gamma_value,
            rationality_value,
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
        for repeat_idx in range(repeats)
        for gamma_idx, gamma_value in enumerate(gamma_s)
        for rationality_idx, rationality_value in enumerate(rationality)
    )

    print(f"Running {n_tasks} simulations with {workers} worker process(es).")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for (
            repeat_idx,
            gamma_idx,
            rationality_idx,
            mean_val,
            var_val,
            ac_time,
            acf,
        ) in tqdm(
            executor.map(
                _measure_environment_heatmap_task,
                tasks,
                chunksize=chunksize,
            ),
            desc=(
                "Collecting equilibrium mean environment, spatial variance, and "
                "temporal autocorrelation"
            ),
            total=n_tasks,
        ):
            mean_environment[repeat_idx, gamma_idx, rationality_idx] = mean_val
            spatial_variance[repeat_idx, gamma_idx, rationality_idx] = var_val
            autocorrelation_time[repeat_idx, gamma_idx, rationality_idx] = ac_time
            acf_sum[gamma_idx, rationality_idx] += acf
            acf_count[gamma_idx, rationality_idx] += 1

    mean_autocorrelation = np.full_like(acf_sum, np.nan)
    valid = acf_count > 0
    mean_autocorrelation[valid] = acf_sum[valid] / acf_count[valid, None]

    peer_pressure_label = (
        "peerconst" if peer_pressure_coeff is not None else "peerrandomised"
    )
    predict_label = "predictive" if neighborhood_prediction_option else "nonpredictive"
    adaptive_label = dynamic_action_preference or "nonadaptive"

    if adaptive_label == "adaptive":
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


def _measure_fourier_heatmap_task(args: tuple):
    (
        base_seed,
        repeat_idx,
        gamma_idx,
        rationality_idx,
        gamma_value,
        rationality_value,
        neighborhood_prediction_option,
        dynamic_action_preference,
        peer_pressure_coeff,
        memory_count,
        width,
        height,
        steps,
    ) = args

    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    seed_seq = np.random.SeedSequence(
        [base_seed, repeat_idx, gamma_idx, rationality_idx]
    )
    rng = np.random.default_rng(seed_seq)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality_value,
        neighb_prediction_option=neighborhood_prediction_option,
        severity_benefit_option=dynamic_action_preference,
        b_1=np.full(width * height, 1.0),
        b_2=peer_pressure_coeff,
        max_storage=steps,
        gamma_s=gamma_value,
    )

    model.run(steps)
    env_timeseries = model.environment[: model.time].mean(axis=1)
    _, power = get_dominant_frequency_and_power(env_timeseries)
    return (repeat_idx, gamma_idx, rationality_idx, power)


def run_measure_fourier_heatmap(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    neighborhood_prediction_option: str | None,
    dynamic_action_preference: str | None,
    peer_pressure_coeff: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
    memory_count: int,
    base_seed: int,
) -> None:
    power_data = np.empty((repeats, len(gamma_s), len(rationality)), dtype=np.float64)

    n_tasks = repeats * len(gamma_s) * len(rationality)
    workers = get_parallel_workers()

    tasks = (
        (
            base_seed,
            repeat_idx,
            gamma_idx,
            rationality_idx,
            gamma_value,
            rationality_value,
            neighborhood_prediction_option,
            dynamic_action_preference,
            peer_pressure_coeff,
            memory_count,
            width,
            height,
            steps,
        )
        for repeat_idx in range(repeats)
        for gamma_idx, gamma_value in enumerate(gamma_s)
        for rationality_idx, rationality_value in enumerate(rationality)
    )

    print(f"Using {workers} worker process(es) for Fourier power measurement.")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for repeat_idx, gamma_idx, rationality_idx, val in tqdm(
            executor.map(_measure_fourier_heatmap_task, tasks, chunksize=1),
            desc="Collecting peak Fourier power",
            total=n_tasks,
        ):
            power_data[repeat_idx, gamma_idx, rationality_idx] = val

    adaptive_label = dynamic_action_preference or "nonadaptive"
    peer_pressure_label = (
        "peerconst" if peer_pressure_coeff is not None else "peerrandomised"
    )
    predict_label = "predictive" if neighborhood_prediction_option else "nonpredictive"

    filename = (
        f"fourier_power_rationality_vs_gamma_s"
        f"_{adaptive_label}_{predict_label}_{peer_pressure_label}"
        f"_{quality_label}_quality.npz"
    )

    np.savez(
        data_dir / filename,
        power=power_data,
        gamma_s=gamma_s,
        rationality=rationality,
    )


def _measure_eq_env_vs_rationality_task(args: tuple):
    (
        repeat_idx,
        rationality_idx,
        rationality_value,
        gamma_s,
        neighborhood_prediction_option,
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

    rng = np.random.default_rng(repeat_idx)
    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality_value,
        simmer_time=1,
        neighb_prediction_option=neighborhood_prediction_option,
        severity_benefit_option=severity_benefit_option,
        radius_option=radius_option,
        max_storage=steps,
        b_1=np.full(width * height, 1.0),
        b_2=b_2,
        gamma_s=gamma_s,
    )
    model.run(steps)

    final_env = model.environment[model.time]
    mean_env = final_env.mean()
    return (repeat_idx, rationality_idx, mean_env)


def run_measure_eq_env_vs_rationality(
    savedir: Path,
    gamma_s: float,
    memory_count_single: int,
    neighb_prediction_option: str | None,
    severity_benefit_option: str | None,
    radius_option: str,
    b_2: npt.NDArray[np.float64] | None,
    env_update_fn_type: str,
    repeats: int,
    min_rationality: float,
    max_rationality: float,
    rationality_steps: int,
    quality_label: str,
    width: int,
    height: int,
    steps: int,
) -> None:
    rationality = np.linspace(min_rationality, max_rationality, rationality_steps)
    results = np.empty((repeats, len(rationality)), dtype=np.float64)

    tasks = [
        (
            repeat_idx,
            rationality_idx,
            rationality_value,
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
        for repeat_idx in range(repeats)
        for rationality_idx, rationality_value in enumerate(rationality)
    ]

    n_tasks = repeats * len(rationality)
    with ProcessPoolExecutor() as executor:
        for repeat_idx, rationality_idx, mean_env in tqdm(
            executor.map(_measure_eq_env_vs_rationality_task, tasks),
            total=n_tasks,
        ):
            results[repeat_idx, rationality_idx] = mean_env

    np.savez(
        savedir / f"eq_env_vs_rationality_{quality_label}_quality.npz",
        mean_environment=results,
        rationality=rationality,
    )


def _measure_eq_env_vs_gamma_task(args: tuple):
    (
        repeat_idx,
        gamma_idx,
        rationality,
        gamma_s,
        b_2,
        width,
        height,
        steps,
    ) = args

    env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    rng = np.random.default_rng(repeat_idx)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality,
        neighb_prediction_option=None,
        max_storage=steps,
        gamma_s=gamma_s,
        b_2=b_2,
    )
    model.run(steps)

    final_env = model.environment[model.time]
    mean_env = final_env.mean()
    return (repeat_idx, gamma_idx, mean_env)


def run_measure_eq_env_vs_gamma(
    savedir: Path,
    quality_label: str,
    rationality: float,
    repeats: int,
    min_gamma_s: float,
    max_gamma_s: float,
    gamma_steps: int,
    b_2: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
) -> None:
    gamma_s_values = np.linspace(min_gamma_s, max_gamma_s, gamma_steps)
    results_env = np.empty((repeats, gamma_steps))

    tasks = [
        (
            repeat_idx,
            gamma_idx,
            rationality,
            gamma_value,
            b_2,
            width,
            height,
            steps,
        )
        for repeat_idx in range(repeats)
        for gamma_idx, gamma_value in enumerate(gamma_s_values)
    ]

    n_tasks = repeats * gamma_steps
    with ProcessPoolExecutor() as executor:
        for repeat_idx, gamma_idx, mean_env in tqdm(
            executor.map(_measure_eq_env_vs_gamma_task, tasks),
            total=n_tasks,
        ):
            results_env[repeat_idx, gamma_idx] = mean_env

    peer_pressure_label = "peerconst" if b_2 is not None else "peerrandomised"
    np.savez(
        savedir
        / f"eq_env_vs_gamma_s_{peer_pressure_label}_{quality_label}_quality.npz",
        mean_environment=results_env,
        gamma_s=gamma_s_values,
    )


def _measure_fourier_vs_gamma_task(args: tuple):
    (
        repeat_idx,
        gamma_idx,
        rationality,
        gamma_s,
        b_2,
        width,
        height,
        steps,
    ) = args

    env_update_fn = piecewise_exponential_update(1, 1, 0.01)
    rng = np.random.default_rng(repeat_idx)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality,
        neighb_prediction_option=None,
        max_storage=steps,
        gamma_s=gamma_s,
        b_2=b_2,
    )
    model.run(steps)

    env_timeseries = model.environment[: model.time].mean(axis=1)
    _, power = get_dominant_frequency_and_power(env_timeseries)
    return (repeat_idx, gamma_idx, power)


def run_measure_fourier_vs_gamma(
    savedir: Path,
    quality_label: str,
    rationality: float,
    repeats: int,
    min_gamma_s: float,
    max_gamma_s: float,
    gamma_steps: int,
    b_2: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
) -> None:
    gamma_s_values = np.linspace(min_gamma_s, max_gamma_s, gamma_steps)
    power_values = np.empty((repeats, gamma_steps), dtype=np.float64)

    tasks = (
        (
            repeat_idx,
            gamma_idx,
            rationality,
            gamma_value,
            b_2,
            width,
            height,
            steps,
        )
        for repeat_idx in range(repeats)
        for gamma_idx, gamma_value in enumerate(gamma_s_values)
    )

    workers = get_parallel_workers()
    n_tasks = repeats * gamma_steps
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for repeat_idx, gamma_idx, power in tqdm(
            executor.map(_measure_fourier_vs_gamma_task, tasks, chunksize=1),
            total=n_tasks,
            desc="Collecting dominant Fourier power",
        ):
            power_values[repeat_idx, gamma_idx] = power

    peer_pressure_label = "peerconst" if b_2 is not None else "peerrandomised"
    np.savez(
        savedir
        / f"fourier_power_vs_gamma_s_{peer_pressure_label}_{quality_label}_quality.npz",
        power=power_values,
        gamma_s=gamma_s_values,
    )


def _measure_timeseries_task(args: tuple):
    (
        repeat_idx,
        rationality_idx,
        rationality_value,
        gamma_s,
        width,
        height,
        steps,
    ) = args

    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = np.random.default_rng(repeat_idx)

    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality_value,
        neighb_prediction_option=None,
        max_storage=steps,
        gamma_s=gamma_s,
        b_2=np.full(width * height, 1.0),
    )
    model.run(steps)

    mean_environment = model.environment[: model.time + 1].mean(axis=1)
    return (repeat_idx, rationality_idx, mean_environment)


def run_measure_timeseries(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: float,
    width: int,
    height: int,
    steps: int,
) -> None:
    mean_environment = np.empty(
        (repeats, len(TIME_SERIES_RATIONALITY_VALUES), steps + 1),
        dtype=np.float64,
    )

    tasks = [
        (
            repeat_idx,
            rationality_idx,
            rationality_value,
            gamma_s,
            width,
            height,
            steps,
        )
        for repeat_idx in range(repeats)
        for rationality_idx, rationality_value in enumerate(
            TIME_SERIES_RATIONALITY_VALUES
        )
    ]

    n_tasks = repeats * len(TIME_SERIES_RATIONALITY_VALUES)
    with ProcessPoolExecutor() as executor:
        for repeat_idx, rationality_idx, val in tqdm(
            executor.map(_measure_timeseries_task, tasks),
            desc="Collecting mean environment time series",
            total=n_tasks,
        ):
            mean_environment[repeat_idx, rationality_idx] = val

    np.savez(
        data_dir
        / f"time_series_mean_env_for_varying_rationality_{quality_label}_quality.npz",
        mean_environment=mean_environment,
        rationality=TIME_SERIES_RATIONALITY_VALUES,
    )


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _steps_at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be >= 2")
    return parsed


def _resolve_heatmap_quality(quick: bool) -> tuple[str, int, int, int]:
    if quick:
        return (
            "low",
            HEATMAP_QUICK_REPEATS,
            HEATMAP_QUICK_GAMMA_S_STEPS,
            HEATMAP_QUICK_RATIONALITY_STEPS,
        )
    return (
        "high",
        HEATMAP_FULL_REPEATS,
        HEATMAP_FULL_GAMMA_S_STEPS,
        HEATMAP_FULL_RATIONALITY_STEPS,
    )


def _run_heatmap(args: argparse.Namespace) -> None:
    quality_label, repeats, gamma_s_steps, rationality_steps = _resolve_heatmap_quality(
        quick=args.quick
    )

    if args.gamma_s_steps is not None:
        gamma_s_steps = args.gamma_s_steps
    if args.rationality_steps is not None:
        rationality_steps = args.rationality_steps

    gamma_s = np.linspace(MIN_GAMMA_S, MAX_GAMMA_S, gamma_s_steps)
    rationality = np.linspace(MIN_RATIONALITY, MAX_RATIONALITY, rationality_steps)

    peer_pressure_coeff = (
        CONST_PEER_PRESSURE_WEIGHT if args.const_peer_pressure else None
    )
    dynamic_action_preference = "adaptive" if args.dynamic_action_preference else None

    if args.metric in ("environment", "both"):
        run_measure_environment_heatmap(
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

    if args.metric in ("fourier", "both"):
        run_measure_fourier_heatmap(
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
            base_seed=args.base_seed,
        )


def _run_curve(args: argparse.Namespace) -> None:
    if args.metric == "eq-env-vs-rationality":
        if args.quick:
            quality_label = "low"
            repeats = EQ_ENV_VS_RATIONALITY_QUICK_REPEATS
            rationality_steps = EQ_ENV_VS_RATIONALITY_QUICK_STEPS
        else:
            quality_label = "high"
            repeats = EQ_ENV_VS_RATIONALITY_FULL_REPEATS
            rationality_steps = EQ_ENV_VS_RATIONALITY_FULL_STEPS

        run_measure_eq_env_vs_rationality(
            savedir=DATA_DIR,
            gamma_s=EQ_ENV_VS_RATIONALITY_GAMMA_S,
            memory_count_single=EQ_ENV_VS_RATIONALITY_MEMORY_SIZE,
            neighb_prediction_option=EQ_ENV_VS_RATIONALITY_NEIGHBORHOOD_PREDICTION,
            severity_benefit_option=EQ_ENV_VS_RATIONALITY_SEVERITY,
            radius_option=EQ_ENV_VS_RATIONALITY_RADIUS,
            b_2=EQ_ENV_VS_RATIONALITY_B2,
            env_update_fn_type=EQ_ENV_VS_RATIONALITY_ENV_UPDATE_FN,
            repeats=repeats,
            min_rationality=EQ_ENV_VS_RATIONALITY_MIN,
            max_rationality=EQ_ENV_VS_RATIONALITY_MAX,
            rationality_steps=rationality_steps,
            quality_label=quality_label,
            width=WIDTH,
            height=HEIGHT,
            steps=N_STEPS,
        )
        return

    if args.metric == "eq-env-vs-gamma":
        if args.quick:
            quality_label = "low"
            repeats = EQ_ENV_VS_GAMMA_QUICK_REPEATS
            gamma_steps = EQ_ENV_VS_GAMMA_QUICK_STEPS
        else:
            quality_label = "high"
            repeats = EQ_ENV_VS_GAMMA_FULL_REPEATS
            gamma_steps = EQ_ENV_VS_GAMMA_FULL_STEPS

        b_2 = CONST_PEER_PRESSURE_WEIGHT if args.const_peer_pressure else None
        run_measure_eq_env_vs_gamma(
            savedir=DATA_DIR,
            quality_label=quality_label,
            repeats=repeats,
            rationality=EQ_ENV_VS_GAMMA_RATIONALITY,
            min_gamma_s=MIN_GAMMA_S,
            max_gamma_s=MAX_GAMMA_S,
            gamma_steps=gamma_steps,
            b_2=b_2,
            width=WIDTH,
            height=HEIGHT,
            steps=EQ_ENV_VS_GAMMA_STEPS,
        )
        return

    if args.metric == "fourier-vs-gamma":
        if args.quick:
            quality_label = "low"
            repeats = FOURIER_VS_GAMMA_QUICK_REPEATS
            gamma_steps = FOURIER_VS_GAMMA_QUICK_STEPS
        else:
            quality_label = "high"
            repeats = FOURIER_VS_GAMMA_FULL_REPEATS
            gamma_steps = FOURIER_VS_GAMMA_FULL_STEPS

        b_2 = CONST_PEER_PRESSURE_WEIGHT if args.const_peer_pressure else None
        run_measure_fourier_vs_gamma(
            savedir=DATA_DIR,
            quality_label=quality_label,
            repeats=repeats,
            rationality=FOURIER_VS_GAMMA_RATIONALITY,
            min_gamma_s=MIN_GAMMA_S,
            max_gamma_s=MAX_GAMMA_S,
            gamma_steps=gamma_steps,
            b_2=b_2,
            width=WIDTH,
            height=HEIGHT,
            steps=FOURIER_VS_GAMMA_STEPS,
        )
        return

    raise ValueError(f"Unsupported curve metric: {args.metric}")


def _run_timeseries(args: argparse.Namespace) -> None:
    if args.metric != "env-vs-rationality":
        raise ValueError(f"Unsupported time-series metric: {args.metric}")

    quality_label = "low" if args.quick else "high"
    repeats = TIME_SERIES_QUICK_REPEATS if args.quick else TIME_SERIES_FULL_REPEATS

    run_measure_timeseries(
        data_dir=DATA_DIR,
        quality_label=quality_label,
        repeats=repeats,
        gamma_s=TIME_SERIES_GAMMA_S,
        width=WIDTH,
        height=HEIGHT,
        steps=TIME_SERIES_STEPS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified measurement runner for heatmap sweeps, 1D curves, and "
            "time-series experiments."
        )
    )
    framework_parsers = parser.add_subparsers(dest="framework", required=True)

    heatmap = framework_parsers.add_parser("heatmap")
    heatmap.add_argument(
        "--metric",
        choices=["environment", "fourier", "both"],
        default="both",
    )
    heatmap.add_argument("--quick", action="store_true")
    heatmap.add_argument("--const-peer-pressure", action="store_true")
    heatmap.add_argument(
        "--dynamic-action-preference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    heatmap.add_argument("--neighborhood-prediction", default=None)
    heatmap.add_argument("--base-seed", type=_non_negative_int, default=0)
    heatmap.add_argument("--variance-last-k", type=int, default=1)
    heatmap.add_argument("--acf-max-lag", type=int, default=200)
    heatmap.add_argument("--gamma-s-steps", type=_steps_at_least_two)
    heatmap.add_argument("--rationality-steps", type=_steps_at_least_two)

    curve = framework_parsers.add_parser("curve")
    curve.add_argument(
        "--metric",
        choices=["eq-env-vs-rationality", "eq-env-vs-gamma", "fourier-vs-gamma"],
        required=True,
    )
    curve.add_argument("--quick", action="store_true")
    curve.add_argument("--const-peer-pressure", action="store_true")

    timeseries = framework_parsers.add_parser("timeseries")
    timeseries.add_argument("--metric", choices=["env-vs-rationality"], required=True)
    timeseries.add_argument("--quick", action="store_true")

    return parser


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.framework == "heatmap":
        _run_heatmap(args)
        return

    if args.framework == "curve":
        _run_curve(args)
        return

    if args.framework == "timeseries":
        _run_timeseries(args)
        return

    parser.error(f"Unsupported framework: {args.framework}")


if __name__ == "__main__":
    run()

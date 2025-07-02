import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from abm_project.batch_run_tools import get_dominant_frequency_and_power
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def measure_peak_frequency_power(args):
    (
        r,
        γi,
        λi,
        γ,
        λ,
        neighb_predict_option,
        dynamic_action_option,
        peer_pressure_coeff,
        memory_count,
        width,
        height,
        steps,
    ) = args
    env_update_fn = piecewise_exponential_update(recovery=1, pollution=1, gamma=0.01)
    rng = np.random.default_rng(r)
    model = VectorisedModel(
        num_agents=width * height,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=λ,
        neighb_prediction_option=neighb_predict_option,
        severity_benefit_option=dynamic_action_option,
        b_1=np.full(width * height, 1.0),
        b_2=peer_pressure_coeff,
        max_storage=steps,
        gamma_s=γ,
    )
    model.run(steps)
    env_timeseries = model.environment[: model.time].mean(axis=1)
    _, power = get_dominant_frequency_and_power(env_timeseries)
    return (r, γi, λi, power)


def main(
    data_dir: Path,
    quality_label: str,
    repeats: int,
    gamma_s: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    neighborhood_prediction_option: str,
    dynamic_action_preference: str,
    peer_pressure_coeff: npt.NDArray[np.float64] | None,
    width: int,
    height: int,
    steps: int,
    memory_count: int,
):
    power_data = np.empty((repeats, len(gamma_s), len(rationality)), dtype=np.float64)

    with ProcessPoolExecutor() as executor:
        futures = []
        for r in range(repeats):
            for γi, γ in enumerate(gamma_s):
                for λi, λ in enumerate(rationality):
                    future = executor.submit(
                        measure_peak_frequency_power,
                        (
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
                        ),
                    )
                    futures.append(future)

        desc = "Collecting peak fourier power for varying rationality, agent adaptation"
        n_tasks = repeats * len(rationality) * len(gamma_s)
        for future in tqdm(as_completed(futures), desc=desc, total=n_tasks):
            (repeat, γi, λi, val) = future.result()
            power_data[repeat, γi, λi] = val

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

    QUICK_REPEATS = 5
    QUICK_GAMMA_S_STEPS = 20
    QUICK_RATIONALITY_STEPS = 20

    FULL_REPEATS = 25
    FULL_GAMMA_S_STEPS = 20
    FULL_RATIONALITY_STEPS = 20

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument("--dynamic-action-preference", action="store_true")
    parser.add_argument("--neighborhood-prediction")
    args = parser.parse_args()

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
    )

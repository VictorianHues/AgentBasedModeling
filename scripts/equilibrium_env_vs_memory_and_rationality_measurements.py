"""Script to investigate cluster formation across varying parameters."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def measure_equilibrium_environment(args):
    (
        r,
        mi,
        λi,
        m,
        λ,
        gamma_s,
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
        memory_count=m,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=λ,
        neighb_prediction_option="linear",
        severity_benefit_option="adaptive",
        max_storage=steps,
        gamma_s=gamma_s,
    )
    model.run(steps)
    final_env = model.environment[model.time]
    mean_env = final_env.mean()
    return (r, mi, λi, mean_env)


def main(
    width: int,
    height: int,
    steps: int,
    agent_adaptation_rate: float,
    min_memory_size: int,
    max_memory_size: int,
    memory_size_step_size: int,
    min_rationality: float,
    max_rationality: float,
    rationality_steps: float,
    n_repeats: int,
    recovery: float,
    pollution: float,
    environment_update_rate: float,
    savedir: Path,
    quality_label: str,
):
    memory_sizes = np.arange(min_memory_size, max_memory_size, memory_size_step_size)
    rationalities = np.linspace(min_rationality, max_rationality, rationality_steps)

    equilibrium_env_results = np.empty(
        (n_repeats, len(memory_sizes), len(rationalities)), dtype=np.float64
    )
    with ProcessPoolExecutor() as executor:
        futures = []
        for r in range(repeats):
            for mi, m in enumerate(memory_sizes):
                for λi, λ in enumerate(rationalities):
                    future = executor.submit(
                        measure_equilibrium_environment,
                        (
                            r,
                            mi,
                            λi,
                            m,
                            λ,
                            agent_adaptation_rate,
                            width,
                            height,
                            steps,
                        ),
                    )
                    futures.append(future)
        desc = (
            "Collecting equilibrium mean environment for varying memory size"
            " and agent adaptation"
        )
        n_tasks = repeats * len(memory_sizes) * len(rationalities)
        for future in tqdm(as_completed(futures), desc=desc, total=n_tasks):
            (repeat, mi, λi, val) = future.result()
            equilibrium_env_results[repeat, mi, λi] = val

    filename = (
        f"equilibrium_env_rationality_vs_memory_gamma_s"
        f"_{str(agent_adaptation_rate).replace('.', '_')}"
        f"_{quality_label}_quality.npz"
    )
    np.savez(
        savedir / filename,
        mean_environment=equilibrium_env_results,
        memory_sizes=memory_sizes,
        rationalities=rationalities,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    # Set model parameters
    WIDTH = 30
    HEIGHT = 30
    STEPS = 1000
    RECOVERY = 1
    POLLUTION = 1
    ENVIRONMENT_UPDATE_RATE = 0.01

    # Experiment parameters
    MIN_MEMORY_SIZE = 10
    MAX_MEMORY_SIZE = 151
    MIN_RATIONALITY = 0.01
    MAX_RATIONALITY = 2.0

    # Quick vs. full (report) experiment parameters
    QUICK_REPEATS = 3
    QUICK_MEMORY_STEP_SIZE = 10
    QUICK_RATIONALITY_STEPS = 10
    FULL_REPEATS = 5
    FULL_MEMORY_STEP_SIZE = 20
    FULL_RATIONALITY_STEPS = 20

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gamma-s", type=str)
    args = parser.parse_args()

    if args.quick:
        repeats = QUICK_REPEATS
        memory_step_size = QUICK_MEMORY_STEP_SIZE
        rationality_steps = QUICK_RATIONALITY_STEPS
        quality_label = "low"
    else:
        repeats = FULL_REPEATS
        memory_step_size = FULL_MEMORY_STEP_SIZE
        rationality_steps = FULL_RATIONALITY_STEPS
        quality_label = "high"

    gamma_s = float(args.gamma_s.replace("_", "."))

    main(
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        agent_adaptation_rate=gamma_s,
        min_memory_size=MIN_MEMORY_SIZE,
        max_memory_size=MAX_MEMORY_SIZE,
        memory_size_step_size=memory_step_size,
        min_rationality=MIN_RATIONALITY,
        max_rationality=MAX_RATIONALITY,
        rationality_steps=rationality_steps,
        n_repeats=repeats,
        recovery=RECOVERY,
        pollution=POLLUTION,
        environment_update_rate=ENVIRONMENT_UPDATE_RATE,
        savedir=DATA_DIR,
        quality_label=quality_label,
    )

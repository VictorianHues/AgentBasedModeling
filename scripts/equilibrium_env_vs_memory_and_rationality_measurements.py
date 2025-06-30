"""Script to investigate cluster formation across varying parameters."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


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
    # Model parameters
    env_update_fn = piecewise_exponential_update(
        recovery=recovery,
        pollution=pollution,
        gamma=environment_update_rate,
    )
    memory_sizes = np.arange(min_memory_size, max_memory_size, memory_size_step_size)
    rationalities = np.linspace(min_rationality, max_rationality, rationality_steps)

    equilibrium_env_results = np.empty(
        (n_repeats, len(memory_sizes), len(rationalities)), dtype=np.float64
    )
    total_iters = n_repeats * len(memory_sizes) * len(rationalities)
    prog_bar = tqdm(
        desc="Measuring equilibrium environment vs. λ and memory-size",
        total=total_iters,
    )
    with prog_bar as bar:
        for r in range(n_repeats):
            for ki, k in enumerate(memory_sizes):
                for λi, λ in enumerate(rationalities):
                    rng = np.random.default_rng(r)
                    model = VectorisedModel(
                        num_agents=width * height,
                        width=width,
                        height=height,
                        memory_count=k,
                        rng=rng,
                        env_update_fn=env_update_fn,
                        rationality=λ,
                        gamma_s=agent_adaptation_rate,
                        neighb_prediction_option="linear",
                        severity_benefit_option="adaptive",
                        max_storage=steps,
                    )
                    model.run(steps)
                    equilibrium_env_results[r, ki, λi] = model.environment[
                        model.time
                    ].mean()
                    bar.update()

    np.savez(
        savedir / f"eq_env_vs_rationality_and_memory_{quality_label}_quality.npz",
        mean_environment=equilibrium_env_results,
        memory_sizes=memory_sizes,
        rationalities=rationalities,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")

    # Set model parameters
    WIDTH = 50
    HEIGHT = 50
    STEPS = 1000
    AGENT_ADAPTATION_RATE = 0.004
    RECOVERY = 1
    POLLUTION = 1
    ENVIRONMENT_UPDATE_RATE = 0.01

    # Experiment parameters
    MIN_MEMORY_SIZE = 10
    MAX_MEMORY_SIZE = 151
    MIN_RATIONALITY = 0.01
    MAX_RATIONALITY = 2.0

    # Quick vs. full (report) experiment parameters
    QUICK_REPEATS = 1
    QUICK_MEMORY_STEP_SIZE = 25
    QUICK_RATIONALITY_STEPS = 5
    FULL_REPEATS = 5
    FULL_MEMORY_STEP_SIZE = 10
    FULL_RATIONALITY_STEPS = 15

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
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

    main(
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        agent_adaptation_rate=AGENT_ADAPTATION_RATE,
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

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tqdm as tqdm

from abm_project import cluster_analysis, metrics
from abm_project.batch_run_tools import get_dominant_frequency_and_power
from abm_project.sensitivity_analysis import sample_parameter_space
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def run_single_simulation(i, repeat, steps, recovery_rate, **kwargs):
    rng = np.random.default_rng(repeat)
    env_update_fn = piecewise_exponential_update(
        recovery=recovery_rate, pollution=1, gamma=0.01
    )
    model = VectorisedModel(rng=rng, env_update_fn=env_update_fn, **kwargs)
    model.run(steps)
    env_mean = np.mean(model.environment[-1])
    action_mean = np.mean(model.action[-1])
    pi_mean = metrics.pluralistic_ignorance(model).mean()
    clusters, _ = cluster_analysis.cluster_given_timestep(
        model, "environment", steps - 1
    )
    peak_freq, power = get_dominant_frequency_and_power(
        model.environment[: model.time].mean(axis=1)
    )

    return i, repeat, env_mean, action_mean, pi_mean, clusters, peak_freq, power


def main(repeats: int, steps: int, n_samples: int, savedir: Path, quality_label: str):
    param_values = sample_parameter_space(n_samples)
    environment_output = np.empty((repeats, len(param_values)))
    action_output = np.empty_like(environment_output)
    pluralistic_ignorance = np.empty_like(environment_output)
    clusters_output = np.empty_like(environment_output)
    peak_frequency = np.empty_like(environment_output)
    dom_frequency_power = np.empty_like(environment_output)

    with ProcessPoolExecutor() as executor:
        futures = []

        for i, (
            width,
            rationality,
            memory_count,
            adaptation_speed,
            radius,
            recovery_rate,
        ) in enumerate(param_values):
            width = int(width)
            memory_count = int(memory_count)
            radius = np.round(radius)
            if radius == 1:
                radius_str = "all"
            elif radius == 0:
                radius_str = "single"
            else:
                raise ValueError("Unsupported radius value")

            kwargs = {
                "num_agents": width * width,
                "width": width,
                "height": width,
                "memory_count": memory_count,
                "rationality": rationality,
                "max_storage": steps,
                "moore": True,
                "simmer_time": 1,
                "neighb_prediction_option": "linear",
                "severity_benefit_option": "adaptive",
                "radius_option": radius_str,
                "prop_pessimistic": 1.0,
                "pessimism_level": 1.0,
                "gamma_s": adaptation_speed,
            }

            for r in range(repeats):
                future = executor.submit(
                    run_single_simulation, i, r, steps, recovery_rate, **kwargs
                )
                futures.append(future)

        for future in tqdm.tqdm(
            as_completed(futures),
            desc="Gathering sensitivity analysis outcome measurements",
            total=repeats * len(param_values),
        ):
            (param_idx, repeat, n_mean, a_mean, pi_mean, clusters, peak_freq, power) = (
                future.result()
            )
            environment_output[repeat, param_idx] = n_mean
            action_output[repeat, param_idx] = a_mean
            pluralistic_ignorance[repeat, param_idx] = pi_mean
            clusters_output[repeat, param_idx] = clusters
            peak_frequency[repeat, param_idx] = peak_freq
            dom_frequency_power[repeat, param_idx] = power

    environment_output = environment_output.mean(axis=0)
    action_output = action_output.mean(axis=0)
    pluralistic_ignorance = pluralistic_ignorance.mean(axis=0)
    clusters_output = clusters_output.mean(axis=0)
    peak_frequency = peak_frequency.mean(axis=0)
    dom_frequency_power = dom_frequency_power.mean(axis=0)

    np.savez(
        savedir
        / f"sensitivity_analysis_outcome_measurements_{quality_label}_quality.npz",
        parameters=param_values,
        mean_environment=environment_output,
        mean_action=action_output,
        pluralistic_ignorance=pluralistic_ignorance,
        cluster_count=clusters_output,
        peak_frequency=peak_frequency,
        dominant_frequency_power=dom_frequency_power,
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    STEPS = 2000

    # Quick-run settings
    QUICK_REPEATS = 1
    QUICK_N_SAMPLES = 128

    # Full (report) settings
    FULL_REPEATS = 5
    FULL_N_SAMPLES = 2048

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        repeats = QUICK_REPEATS
        n_samples = QUICK_N_SAMPLES
        quality_label = "low"
    else:
        repeats = FULL_REPEATS
        n_samples = FULL_N_SAMPLES
        quality_label = "high"

    main(repeats, STEPS, n_samples, DATA_DIR, quality_label)

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tqdm as tqdm
from SALib.sample import sobol as sobol_sample

from abm_project import cluster_analysis, metrics
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


# get parameter values
def problem():
    problem = {
        "num_vars": 6,
        "names": [
            "width",
            "rationality",
            "memory_count",
            "adaptation_speed",
            "radius",
            "recovery_rate",
        ],
        "bounds": [
            [5, 50],
            [0, 10],
            [2, 10],
            [0.001, 0.05],
            [0, 1],
            [0.5, 2.0],
        ],
    }
    return problem


def sample_parameter_space(n: int):
    param_values = sobol_sample.sample(problem(), n, calc_second_order=False)
    return param_values


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

    return i, repeat, env_mean, action_mean, pi_mean, clusters


def main(repeats: int, steps: int, n_samples: int, savedir: Path):
    param_values = sample_parameter_space(n_samples)
    environment_output = np.empty((repeats, len(param_values)))
    action_output = np.empty_like(environment_output)
    pluralistic_ignorance = np.empty_like(environment_output)
    clusters_output = np.empty_like(environment_output)

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
            as_completed(futures), total=repeats * len(param_values)
        ):
            param_idx, repeat, n_mean, a_mean, pi_mean, clusters = future.result()
            environment_output[repeat, param_idx] = n_mean
            action_output[repeat, param_idx] = a_mean
            pluralistic_ignorance[repeat, param_idx] = pi_mean
            clusters_output[repeat, param_idx] = clusters

    environment_output = environment_output.mean(axis=0)
    action_output = action_output.mean(axis=0)
    pluralistic_ignorance = pluralistic_ignorance.mean(axis=0)
    clusters_output = clusters_output.mean(axis=0)

    np.savez(
        savedir / "sensitivity_analysis_outcome_measurements.npz",
        parameters=param_values,
        mean_environment=environment_output,
        mean_action=action_output,
        pluralistic_ignorance=pluralistic_ignorance,
        cluster_count=clusters_output,
    )


if __name__ == "__main__":
    REPEATS = 1
    STEPS = 1000
    N_SAMPLES = 256
    DATA_DIR = Path("data")
    main(REPEATS, STEPS, N_SAMPLES, DATA_DIR)

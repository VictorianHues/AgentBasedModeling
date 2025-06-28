"""Functions to measure model observables."""

import numpy as np
import numpy.typing as npt

from abm_project.vectorised_model import VectorisedModel


def pluralistic_ignorance(
    model: VectorisedModel, k: int = 1
) -> npt.NDArray[np.float64]:
    """Measure average pluralistic ignorance.

    Pluralistic ignorance is a phenomenon which occurs when individuals underestimate
    public support for a particular action, leading them to behave in a manner which
    does not reflect their own beliefs, even when true public support is high.

    We measure pluralistic ignorance in the agent-based climate mitigation model by
    calculating, for each agent in the final k simulation timesteps, the difference
    between their normalised average preference for cooperation and their probability
    of cooperation.

    Args:
        model: A VectorisedModel which has been run for at least k timesteps.
        k: Number of timesteps to include in measurement.

    Returns:
        A 1D Numpy array containing the measured pluralistic ignorance for each agent.
    """
    if model.time < k:
        raise ValueError("Model must be run for at least `k` timesteps.")

    start = (model.time - k) + 1
    stop = model.time + 1
    preference = (model.s[start:stop] / 4).mean(axis=0)
    probability = ((model.action[start:stop] + 1) / 2).mean(axis=0)
    return preference - probability

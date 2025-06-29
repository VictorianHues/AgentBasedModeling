"""Functions to measure model observables."""

import numpy as np
import numpy.typing as npt

from abm_project.vectorised_model import VectorisedModel


def pluralistic_ignorance(
    model: VectorisedModel,
) -> npt.NDArray[np.float64]:
    r"""Measure agents' pluralistic ignorance at the end of simulation.

    Pluralistic ignorance is a phenomenon which occurs when individuals underestimate
    public support for a particular action, leading them to behave in a manner which
    does not reflect their own beliefs, even when true public support is high.

    To measure an agent's pluralistic ignorance at the end of simulation, we consider
    their expected actions:

    1. Under perceived social norms, :math:`\mathbb{E}[a_i]_\text{perceived}`
    2. In absence of social norms, :math:`\mathbb{E}[a_i]_\text{individual}`  
    3. When observing their neighbors' true preferences, 
        :math:`\mathbb{E}[a_i]_\text{true}`

    An agent :math:`i`'s pluralistic ignorance is calculated as:

    .. math::
        
        \psi_i = \max\{0, \
        |\mathbb{E}[a_i]_\text{perceived} - \mathbb{E}[a_i]_\text{individual}| - \
        |\mathbb{E}[a_i]_\text{true} - \mathbb{E}[a_i]_\text{individual}|\}

    i.e., it is large when knowing the true social norm would allow an agent to behave
    in a manner more consistent with their individual preferences.

    Args:
        model: A VectorisedModel which has been run for at least k timesteps.

    Returns:
        A 1D Numpy array containing the measured pluralistic ignorance for each agent.
    """
    # Calculate true expected actions
    p_true = model.action_probabilities()
    m_true = p_true[1] - p_true[0]

    # Calculate expected actions conditional on observing neighbor preferences
    model.neighb_prediction_option = "true_pref"
    p_cond = model.action_probabilities()
    m_cond = p_cond[1] - p_cond[0]

    # Expected actions without social norms
    model.b[0] = 1
    model.b[1] = 0
    p_individual = model.action_probabilities()
    m_individual = p_individual[1] - p_individual[0]

    # Pluralistic ignorance if agents would act closer to own pref if social norms known
    true_dist = np.abs(m_true - m_individual)
    cond_dist = np.abs(m_cond - m_individual)

    improvement = true_dist - cond_dist
    improvement[improvement < 0] = 0.0
    return improvement

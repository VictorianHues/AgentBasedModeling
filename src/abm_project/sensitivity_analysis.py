"""ABM model sensitivity analysis utilities."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from SALib.analyze import pawn, sobol
from SALib.sample import sobol as sobol_sample

PARAMETER_NAMES = [
    "Grid length",
    "Rationality",
    "Memory size",
    "Adaptation rate",
    "Neighborhood radius",
    "Recovery rate",
]

OUTCOME_NAMES = [
    "Mean environment",
    "Mean action",
    "Pluralistic ignorance",
    "Cluster count",
    "Peak frequency",
    "Dominant frequency power",
]

PROBLEM = {
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


@dataclass
class PawnIndices:
    """Median Pawn indices for each outcome variable."""

    mean_environment: npt.NDArray[np.float64]
    mean_action: npt.NDArray[np.float64]
    pluralistic_ignorance: npt.NDArray[np.float64]
    cluster_count: npt.NDArray[np.float64]
    peak_frequency: npt.NDArray[np.float64]
    dominant_frequency_power: npt.NDArray[np.float64]

    def stack(self) -> npt.NDArray[np.float64]:
        """Get all indices in a 2D Numpy array.

        Returns:
            2D Numpy array with shape (outcomes, parameters).
        """
        return np.asarray(
            [
                self.mean_environment,
                self.mean_action,
                self.pluralistic_ignorance,
                self.cluster_count,
                self.peak_frequency,
                self.dominant_frequency_power,
            ]
        )


@dataclass
class SobolOrderIndices:
    """Sobol sensitivity index and confidence bound."""

    index: npt.NDArray[np.float64]
    confidence: npt.NDArray[np.float64]


@dataclass
class OutcomeSobolIndices:
    """First- and total-order Sobol indices for a given outcome variable."""

    first_order: SobolOrderIndices
    total_order: SobolOrderIndices


@dataclass
class SobolIndices:
    """First- and total-order Sobol indices for each outcome variable."""

    pluralistic_ignorance: OutcomeSobolIndices

    def to_list(self) -> list[OutcomeSobolIndices]:
        """Get outcome Sobol indices in a list, for iteration purposes."""
        return [
            self.pluralistic_ignorance,
        ]


def sample_parameter_space(n: int) -> npt.NDArray[np.float64]:
    """Sample parameter values for sensitivity analysis.

    Args:
        n: Number of samples to generate.

    Returns:
        2D Numpy array of parameter value samples.
    """
    return sobol_sample.sample(PROBLEM, n, calc_second_order=False)


def sobol_analysis(
    pluralistic_ignorance: npt.NDArray[np.float64],
) -> SobolIndices:
    """Compute Sobol indices for each outcome variable.

    Supported outcome variables:
    - Mean environment         (x)
    - Mean action              (x)
    - Pluralistic ignorance
    - Cluster count            (x)

    (x) indicates that an outcome is not included due to distributional problems.

    Args:
        pluralistic_ignorance: For each parameter sample, the average agent pluralistic
            ignorance at the end of simulation.

    Returns:
        A SobolIndices object with the first-order and total-order Sobol indices for
        each varied parameter.
    """

    def compute_sobol_indices(outcome: npt.NDArray[np.float64]):
        result = sobol.analyze(
            PROBLEM,
            Y=outcome,
            calc_second_order=False,
        )

        return OutcomeSobolIndices(
            first_order=SobolOrderIndices(
                index=result["S1"],
                confidence=result["S1_conf"],
            ),
            total_order=SobolOrderIndices(
                index=result["ST"],
                confidence=result["ST_conf"],
            ),
        )

    return SobolIndices(
        pluralistic_ignorance=compute_sobol_indices(pluralistic_ignorance),
    )


def pawn_analysis(
    parameters: npt.NDArray[np.float64],
    mean_environment: npt.NDArray[np.float64],
    mean_action: npt.NDArray[np.float64],
    pluralistic_ignorance: npt.NDArray[np.float64],
    cluster_count: npt.NDArray[np.float64],
    peak_frequency: npt.NDArray[np.float64],
    dominant_frequency_power: npt.NDArray[np.float64],
) -> PawnIndices:
    """Compute median Pawn indices for each outcome variable.

    Supported outcome variables:
    - Mean environment
    - Mean action
    - Pluralistic ignorance
    - Cluster count
    - Peak (fourier) frequency
    - Dominant frequency power

    Args:
        parameters: Parameter values used to generate the outcome measurements
        mean_environment: For each parameter sample, the average environment state
            at the end of simulation.
        mean_action: For each parameter sample, the average agent action at the end of
            simulation.
        pluralistic_ignorance: For each parameter sample, the average agent pluralistic
            ignorance at the end of simulation.
        cluster_count: For each parameter sample, the total number of environment
            clusters at the end of simulation.
        peak_frequency: For each parameter sample, the maximum power frequency observed
            in the mean environment state.
        dominant_frequency_power: The power of the dominant fourier frequency for each
            parameter sample.

    Returns:
        A PawnIndices object with the median Pawn index for each varied parameter.
    """

    def compute_median_indices(outcome: npt.NDArray[np.float64]):
        return pawn.analyze(
            PROBLEM,
            X=parameters,
            Y=outcome,
        )["median"]

    return PawnIndices(
        mean_environment=compute_median_indices(mean_environment),
        mean_action=compute_median_indices(mean_action),
        pluralistic_ignorance=compute_median_indices(pluralistic_ignorance),
        cluster_count=compute_median_indices(cluster_count),
        peak_frequency=compute_median_indices(peak_frequency),
        dominant_frequency_power=compute_median_indices(dominant_frequency_power),
    )

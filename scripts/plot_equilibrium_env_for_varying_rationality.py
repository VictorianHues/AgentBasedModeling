import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def main(
    mean_environment: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    savedir: Path,
    quality_label: str = "low",
    # gamma_s: float = 0.01,
    # memory_count_single: int = 10,
    # neighb_prediction_option=None,
    # severity_benefit_option=None,
    # radius_option="single",
    # b_2=None,
    # env_update_fn_type="linear",
    # repeats: int = 50,
    # min_rationality: float = 0.0,
    # max_rationality: float = 5.0,
):
    fig, ax = plt.subplots(
        figsize=(3.5, 2),
        constrained_layout=True,
    )

    # Plot mean environment at steady state
    repeats = mean_environment.shape[0]
    mean = mean_environment.mean(axis=0)
    std = mean_environment.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)
    ax.plot(rationality, mean, label="Mean")
    ax.fill_between(rationality, mean - ci, mean + ci, alpha=0.3)

    # Indicate point of maximum absolute derivative
    dmean = np.gradient(mean, rationality)
    idx_steepest = np.argmax(np.abs(dmean))
    ax.plot(rationality[idx_steepest], mean[idx_steepest], "ro", label="Steepest point")
    ax.annotate(
        f"Steepest\n($\\lambda$={rationality[idx_steepest]:.2f})",
        xy=(rationality[idx_steepest], mean[idx_steepest]),
        xytext=(rationality[idx_steepest] + 0.2, mean[idx_steepest]),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=8,
        ha="left",
    )

    ax.set_xlabel(r"Rationality ($\lambda$)")
    ax.set_ylabel(r"Equilibrium mean environment ($\overline{n}^*$)")
    ax.set_ylim(0, 1)
    ax.set_xlim(rationality.min(), rationality.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()

    fig.savefig(
        savedir / f"equilibrium_env_vs_rationality_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    data = np.load(DATA_DIR / f"eq_env_vs_rationality_{quality_label}_quality.npz")
    main(
        mean_environment=data["mean_environment"],
        rationality=data["rationality"],
        savedir=FIGURES_DIR,
        quality_label=quality_label,
    )

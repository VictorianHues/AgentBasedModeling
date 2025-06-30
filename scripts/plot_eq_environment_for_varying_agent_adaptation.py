import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def main(
    mean_environment: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    savedir: Path,
    quality_label: str = "low",
):
    fig, ax = plt.subplots(figsize=(3.5, 2), constrained_layout=True)

    repeats = mean_environment.shape[0]
    mean = mean_environment.mean(axis=0)
    std = mean_environment.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)
    ax.plot(gamma_s, mean, label="Mean")
    ax.fill_between(gamma_s, mean - ci, mean + ci, alpha=0.3)

    # Annotate point with steepest gradient
    dmean = np.gradient(mean, gamma_s)
    idx_steepest = np.argmax(np.abs(dmean))
    ax.plot(
        gamma_s[idx_steepest],
        mean[idx_steepest],
        "ro",
        label="Steepest point",
    )
    ax.annotate(
        f"Steepest\n($\\gamma_s$={gamma_s[idx_steepest]:.4f})",
        xy=(gamma_s[idx_steepest], mean[idx_steepest]),
        xytext=(gamma_s[idx_steepest] + 0.002, mean[idx_steepest]),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=10,
        ha="left",
    )

    ax.set_xlabel(r"Support update rate ($\gamma_s$)")
    ax.set_ylabel(r"Equilibrium mean environment ($\overline{n}^*$)")
    ax.set_ylim(0, 1)
    ax.set_xlim(gamma_s.min(), gamma_s.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.savefig(
        savedir / f"equilibrium_env_vs_gamma_s_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    data = np.load(DATA_DIR / f"eq_env_vs_gamma_s_{quality_label}_quality.npz")
    main(
        mean_environment=data["mean_environment"],
        gamma_s=data["gamma_s"],
        savedir=FIGURES_DIR,
        quality_label=quality_label,
    )

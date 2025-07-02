import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def main(
    mean_environment: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    quality_label: str,
    savedir: Path | None = None,
):
    # Plot mean environment at steady state
    fig, ax = plt.subplots(
        figsize=(3.5, 2),
        constrained_layout=True,
    )

    repeats, _, steps = mean_environment.shape
    t = np.arange(steps)
    for λi, λ in enumerate(rationality):
        mean = mean_environment[:, λi].mean(axis=0)
        std = mean_environment[:, λi].std(axis=0, ddof=1)
        ci = 1.97 * std / np.sqrt(repeats)
        ax.plot(t, mean, label=f"$\\lambda = {λ:.2f}$")
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.3)

    ax.set_ylabel("Mean environment")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(
        savedir / f"time_series_mean_env_by_rationality_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    data = np.load(
        DATA_DIR
        / f"time_series_mean_env_for_varying_rationality_{quality_label}_quality.npz"
    )

    main(
        mean_environment=data["mean_environment"],
        rationality=data["rationality"],
        quality_label=quality_label,
        savedir=FIGURES_DIR,
    )

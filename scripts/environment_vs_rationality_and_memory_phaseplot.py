"""Plot phaseplot for equilibrium environment with varying rationality/memory size."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def main(
    mean_environment: npt.NDArray[np.float64],
    memory_sizes: npt.NDArray[np.float64],
    rationalities: npt.NDArray[np.float64],
    savepath: Path,
):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    cbar = ax.imshow(
        mean_environment,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[
            rationalities[0],
            rationalities[-1],
            memory_sizes[0],
            memory_sizes[-1],
        ],
        vmax=1.0,
        vmin=0.0,
    )
    fig.colorbar(cbar, ax=ax, label="Mean environment")
    ax.set_xlabel(r"Rationality $\lambda$")
    ax.set_ylabel(r"Memory size")
    ax.set_xticks(np.linspace(0, rationalities.max(), 5))
    ax.set_yticks(memory_sizes[::2])

    fig.savefig(
        savepath,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gamma-s", type=str)
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"
    filename = (
        f"equilibrium_env_rationality_vs_memory_gamma_s"
        f"_{args.gamma_s}"
        f"_{quality_label}_quality"
    )
    results = np.load(DATA_DIR / f"{filename}.npz")
    main(
        mean_environment=results["mean_environment"].mean(axis=0),
        memory_sizes=results["memory_sizes"],
        rationalities=results["rationalities"],
        savepath=FIGURES_DIR / f"{filename}.pdf",
    )

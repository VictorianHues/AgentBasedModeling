import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns


def main(
    savedir: Path,
    power: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    filename: str,
):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    vmin, vmax = 0, 30_000
    x_tick_indices = np.arange(0, len(rationality), max(1, len(rationality) // 4))
    y_tick_indices = np.arange(0, len(gamma_s), max(1, len(gamma_s) // 4))

    x_tick_labels = [f"{rationality[i]:.1f}" for i in x_tick_indices]
    y_tick_labels = [f"{gamma_s[i]:.3f}" for i in y_tick_indices]

    sns.heatmap(
        power.mean(axis=0),
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        cbar_kws={"label": "Power at dominant frequency", "shrink": 0.7, "pad": 0.02},
        ax=ax,
        square=False,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(r"Rationality ($\lambda$)", fontsize=9)
    ax.set_ylabel(r"Support Update Rate ($\gamma_s$)", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(savedir / filename, bbox_inches="tight")


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument("--dynamic-action-preference", action="store_true")
    parser.add_argument("--neighborhood-prediction")
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    peer_pressure_label = "peerrandomised"
    if args.const_peer_pressure:
        peer_pressure_label = "peerconst"

    dynamic_action_preference = None
    if args.dynamic_action_preference:
        dynamic_action_preference = "adaptive"

    adaptive_label = dynamic_action_preference or "nonadaptive"
    predict_label = "predictive" if args.neighborhood_prediction else "nonpredictive"
    filename = (
        f"fourier_power_rationality_vs_gamma_s"
        f"_{adaptive_label}_{predict_label}_{peer_pressure_label}"
        f"_{quality_label}_quality"
    )
    data = np.load(DATA_DIR / f"{filename}.npz")

    main(
        savedir=FIGURES_DIR,
        power=data["power"],
        rationality=data["rationality"],
        gamma_s=data["gamma_s"],
        filename=f"{filename}.pdf",
    )

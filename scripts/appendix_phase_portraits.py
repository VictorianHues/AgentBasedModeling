from pathlib import Path

import matplotlib.pyplot as plt

from abm_project.plotting import plot_phase_portrait


def main():
    RECOVERY = 1
    POLLUTION = 1
    RESULTS_DIR = Path("results/figures")

    c_vals = (0.5, 0.75)
    λ_vals = (1.0 / (2 * 0.75), 1.0)

    fig, axes = plt.subplots(
        2, 2, figsize=(3.5, 3.5), constrained_layout=True, sharex=True, sharey=True
    )
    for ci, c in enumerate(c_vals):
        for λi, λ in enumerate(λ_vals):
            ax = axes[ci, λi]
            plot_phase_portrait(
                c=c, rationality=λ, recovery=RECOVERY, pollution=POLLUTION, ax=ax
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(f"$w={c:.2f}$, $\\lambda={λ:.2f}$")
            ax.set_aspect(1 / 2)

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel(r"Environment ($n$)")
    fig.supylabel(r"Expected action ($m$)")

    fig.savefig(RESULTS_DIR / "appendix_phase_portraits.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

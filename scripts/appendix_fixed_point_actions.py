from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from abm_project import configure_mpl


def main():
    RESULTS_DIR = Path("results/figures")

    s_vals = (2, 3)

    params = [
        (0.5, 1.0),  # (c, λ)
        (0.6, 1.5),
    ]

    fig, axes = plt.subplots(
        nrows=2,
        figsize=(3.5, 3),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    m = np.linspace(-1, 1, 101)

    for si, (s, ax) in enumerate(zip(s_vals, axes.flatten(), strict=True)):
        ax.plot(m, m, linestyle="dashed", color="grey")
        for c, λ in params:
            rhs = np.tanh(λ * (1 - c) * (s - 2) + 2 * λ * c * m)
            ax.plot(
                m,
                rhs,
                label=f"$w={c:.2f}$, $\\lambda={λ:.2f}$" if si == 0 else None,
            )
        ax.set_title(f"$s={s:.1f}$")

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    fig.supxlabel(r"Expected action ($m$)")
    fig.supylabel(r"$\tanh(\lambda (1 - w)(s - 2) + 2\lambda w m)$")

    fig.legend(
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0, 0, 0),
        handlelength=0.5,
        labelspacing=1,
        columnspacing=0.75,
        frameon=False,
    )

    fig.savefig(
        RESULTS_DIR / "appendix_fixed_point_mean_action.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    configure_mpl()
    main()

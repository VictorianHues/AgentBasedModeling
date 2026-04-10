import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from abm_project.batch_run_tools import estimate_autocorrelation_time

HEATMAP_FONT_FAMILY = "DejaVu Sans"
HEATMAP_LABEL_FONT_SIZE = 9
HEATMAP_TICK_FONT_SIZE = 8


def main(
    savedir: Path,
    values: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    filename: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    x_tick_indices = np.arange(0, len(rationality), max(1, len(rationality) // 4))
    y_tick_indices = np.arange(0, len(gamma_s), max(1, len(gamma_s) // 4))

    x_tick_labels = [f"{rationality[i]:.1f}" for i in x_tick_indices]
    y_tick_labels = [f"{gamma_s[i]:.3f}" for i in y_tick_indices]

    sns.heatmap(
        values.mean(axis=0),
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        cbar_kws={"label": cbar_label, "shrink": 0.7, "pad": 0.02},
        ax=ax,
        square=False,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(
        x_tick_labels,
        fontsize=HEATMAP_TICK_FONT_SIZE,
        fontfamily=HEATMAP_FONT_FAMILY,
    )
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(
        y_tick_labels,
        fontsize=HEATMAP_TICK_FONT_SIZE,
        fontfamily=HEATMAP_FONT_FAMILY,
    )

    ax.set_xlabel(
        r"Rationality ($\lambda$)",
        fontsize=HEATMAP_LABEL_FONT_SIZE,
        fontfamily=HEATMAP_FONT_FAMILY,
    )
    ax.set_ylabel(
        r"Support Update Rate ($\gamma_s$)",
        fontsize=HEATMAP_LABEL_FONT_SIZE,
        fontfamily=HEATMAP_FONT_FAMILY,
    )
    ax.tick_params(axis="both", which="major", labelsize=HEATMAP_TICK_FONT_SIZE)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=HEATMAP_TICK_FONT_SIZE)
    cbar.ax.set_ylabel(
        cbar_label,
        fontsize=HEATMAP_LABEL_FONT_SIZE,
        fontfamily=HEATMAP_FONT_FAMILY,
    )

    fig.savefig(savedir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_variance_slice(
    savedir: Path,
    spatial_variance: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    lambda_value: float,
    filename: str,
):
    repeats = spatial_variance.shape[0]
    lambda_idx = int(np.argmin(np.abs(rationality - lambda_value)))

    mean = spatial_variance[:, :, lambda_idx].mean(axis=0)
    std = spatial_variance[:, :, lambda_idx].std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    ax.plot(gamma_s, mean, color="black", label="Mean variance")
    ax.fill_between(gamma_s, mean - ci, mean + ci, alpha=0.3)

    peak_idx = int(np.nanargmax(mean))
    ax.plot(gamma_s[peak_idx], mean[peak_idx], "ro", label="Peak")
    ax.annotate(
        f"Peak\n($\\gamma_s$={gamma_s[peak_idx]:.4f})",
        xy=(gamma_s[peak_idx], mean[peak_idx]),
        xytext=(gamma_s[peak_idx], mean[peak_idx] * 1.05),
        fontsize=8,
        color="red",
        ha="left",
    )

    ax.set_xlabel(r"Support Update Rate ($\gamma_s$)")
    ax.set_ylabel("Spatial variance")
    ax.set_title(rf"Variance Slice ($\lambda$={rationality[lambda_idx]:.2f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(savedir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_autocorrelation_time_slice(
    savedir: Path,
    autocorrelation_time: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    lambda_value: float,
    filename: str,
):
    repeats = autocorrelation_time.shape[0]
    lambda_idx = int(np.argmin(np.abs(rationality - lambda_value)))

    mean = autocorrelation_time[:, :, lambda_idx].mean(axis=0)
    std = autocorrelation_time[:, :, lambda_idx].std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    ax.plot(gamma_s, mean, color="black", label="Mean autocorrelation time")
    ax.fill_between(gamma_s, mean - ci, mean + ci, alpha=0.3)

    peak_idx = int(np.nanargmax(mean))
    ax.plot(gamma_s[peak_idx], mean[peak_idx], "ro", label="Peak")
    ax.annotate(
        f"Peak\n($\\gamma_s$={gamma_s[peak_idx]:.4f})",
        xy=(gamma_s[peak_idx], mean[peak_idx]),
        xytext=(gamma_s[peak_idx], mean[peak_idx] * 1.05),
        fontsize=8,
        color="red",
        ha="left",
    )

    ax.set_xlabel(r"Support Update Rate ($\gamma_s$)")
    ax.set_ylabel("Autocorrelation time")
    ax.set_title(
        rf"Autocorrelation-Time Slice ($\lambda$={rationality[lambda_idx]:.2f})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(savedir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_selected_acf(
    savedir: Path,
    mean_autocorrelation: npt.NDArray[np.float64],
    acf_lags: npt.NDArray[np.int64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    lambda_value: float,
    gamma_value: float,
    filename: str,
):
    lambda_idx = int(np.argmin(np.abs(rationality - lambda_value)))
    gamma_idx = int(np.argmin(np.abs(gamma_s - gamma_value)))
    acf = mean_autocorrelation[gamma_idx, lambda_idx]
    tau = estimate_autocorrelation_time(acf)

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    ax.plot(acf_lags, acf, color="black", label="Mean ACF")
    ax.axhline(np.exp(-1), color="gray", linestyle="--", linewidth=0.8)
    if np.isfinite(tau):
        ax.axvline(tau, color="red", linestyle="--", linewidth=1, label=r"$\tau_c$")
        ax.text(
            0.97,
            0.95,
            f"$\\tau_c$={tau:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
        )

    ax.set_xlabel("Lag")
    ax.set_ylabel("C(lag)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        rf"ACF of Mean Environment ($\lambda$={rationality[lambda_idx]:.2f}, "
        rf"$\gamma_s$={gamma_s[gamma_idx]:.4f})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(savedir / filename, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--const-peer-pressure", action="store_true")
    parser.add_argument(
        "--dynamic-action-preference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use adaptive action preference dynamics. Disabled by default; pass "
            "--dynamic-action-preference to load adaptive heatmaps."
        ),
    )
    parser.add_argument("--neighborhood-prediction", default=None)
    parser.add_argument("--variance-slice-lambda", type=float)
    parser.add_argument("--autocorr-slice-lambda", type=float)
    parser.add_argument("--acf-lambda", type=float)
    parser.add_argument("--acf-gamma", type=float)
    args = parser.parse_args()

    quality_label = "low" if args.quick else "high"

    peer_pressure_label = "peerrandomised"
    if args.const_peer_pressure:
        peer_pressure_label = "peerconst"

    dynamic_action_preference = None
    if args.dynamic_action_preference:
        dynamic_action_preference = "adaptive"

    adaptive_label = dynamic_action_preference or "nonadaptive"
    mode_label = "" if adaptive_label == "adaptive" else f"{adaptive_label}_"

    predict_label = "predictive" if args.neighborhood_prediction else "nonpredictive"
    if adaptive_label == "adaptive":
        # Preserve existing filename pattern for backward compatibility.
        filename = (
            f"equilibrium_env_rationality_vs_gamma_s"
            f"_{predict_label}_{peer_pressure_label}"
            f"_{quality_label}_quality"
        )
    else:
        filename = (
            f"equilibrium_env_rationality_vs_gamma_s"
            f"_{adaptive_label}_{predict_label}_{peer_pressure_label}"
            f"_{quality_label}_quality"
        )
    data = np.load(DATA_DIR / f"{filename}.npz")

    main(
        savedir=FIGURES_DIR,
        values=data["mean_environment"],
        rationality=data["rationality"],
        gamma_s=data["gamma_s"],
        filename=f"{filename}.pdf",
        cbar_label="Mean environmental status",
        vmin=0,
        vmax=1,
    )

    variance_filename = (
        f"spatial_variance_rationality_vs_gamma_s"
        f"_{mode_label}{predict_label}_{peer_pressure_label}"
        f"_{quality_label}_quality.pdf"
    )
    main(
        savedir=FIGURES_DIR,
        values=data["spatial_variance"],
        rationality=data["rationality"],
        gamma_s=data["gamma_s"],
        filename=variance_filename,
        cbar_label="Spatial variance",
        vmin=0,
        vmax=float(np.nanmax(data["spatial_variance"].mean(axis=0))),
    )

    ac_time_filename = (
        f"autocorrelation_time_rationality_vs_gamma_s"
        f"_{mode_label}{predict_label}_{peer_pressure_label}"
        f"_{quality_label}_quality.pdf"
    )
    main(
        savedir=FIGURES_DIR,
        values=data["autocorrelation_time"],
        rationality=data["rationality"],
        gamma_s=data["gamma_s"],
        filename=ac_time_filename,
        cbar_label="Autocorrelation time",
        vmin=0,
        vmax=float(np.nanmax(data["autocorrelation_time"].mean(axis=0))),
    )

    if args.variance_slice_lambda is not None:
        slice_filename = (
            f"spatial_variance_vs_gamma_s_lambda{args.variance_slice_lambda:.2f}"
            f"_{mode_label}{predict_label}_{peer_pressure_label}_{quality_label}_quality.pdf"
        )
        plot_variance_slice(
            savedir=FIGURES_DIR,
            spatial_variance=data["spatial_variance"],
            rationality=data["rationality"],
            gamma_s=data["gamma_s"],
            lambda_value=args.variance_slice_lambda,
            filename=slice_filename,
        )

    if args.autocorr_slice_lambda is not None:
        ac_slice_filename = (
            f"autocorrelation_time_vs_gamma_s_lambda{args.autocorr_slice_lambda:.2f}"
            f"_{mode_label}{predict_label}_{peer_pressure_label}_{quality_label}_quality.pdf"
        )
        plot_autocorrelation_time_slice(
            savedir=FIGURES_DIR,
            autocorrelation_time=data["autocorrelation_time"],
            rationality=data["rationality"],
            gamma_s=data["gamma_s"],
            lambda_value=args.autocorr_slice_lambda,
            filename=ac_slice_filename,
        )

    if args.acf_lambda is not None and args.acf_gamma is not None:
        acf_filename = (
            f"acf_mean_env_lambda{args.acf_lambda:.2f}_gamma{args.acf_gamma:.4f}"
            f"_{mode_label}{predict_label}_{peer_pressure_label}_{quality_label}_quality.pdf"
        )
        plot_selected_acf(
            savedir=FIGURES_DIR,
            mean_autocorrelation=data["mean_autocorrelation"],
            acf_lags=data["acf_lags"],
            rationality=data["rationality"],
            gamma_s=data["gamma_s"],
            lambda_value=args.acf_lambda,
            gamma_value=args.acf_gamma,
            filename=acf_filename,
        )

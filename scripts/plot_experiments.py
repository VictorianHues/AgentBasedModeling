import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from abm_project.batch_run_tools import estimate_autocorrelation_time

DATA_DIR = Path("data")
FIGURES_DIR = Path("results/figures")

HEATMAP_FONT_FAMILY = "DejaVu Sans"
HEATMAP_LABEL_FONT_SIZE = 9
HEATMAP_TICK_FONT_SIZE = 8


def plot_heatmap(
    savedir: Path,
    values: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    filename: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
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
) -> None:
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
) -> None:
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
) -> None:
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


def plot_eq_env_vs_rationality(
    mean_environment: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    savedir: Path,
    quality_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2), constrained_layout=True)

    repeats = mean_environment.shape[0]
    mean = mean_environment.mean(axis=0)
    std = mean_environment.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)

    ax.plot(rationality, mean, label="Mean")
    ax.fill_between(rationality, mean - ci, mean + ci, alpha=0.3)

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
    plt.close(fig)


def plot_curve_vs_gamma(
    values: npt.NDArray[np.float64],
    gamma_s: npt.NDArray[np.float64],
    ylabel: str,
    filename_prefix: str,
    savedir: Path,
    quality_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2), constrained_layout=True)

    repeats = values.shape[0]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    ci = 1.97 * std / np.sqrt(repeats)

    ax.plot(gamma_s, mean, label="Mean")
    ax.fill_between(gamma_s, mean - ci, mean + ci, alpha=0.3)

    dmean = np.gradient(mean, gamma_s)
    idx_steepest = np.argmax(np.abs(dmean))
    ax.plot(gamma_s[idx_steepest], mean[idx_steepest], "ro", label="Steepest point")
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
    ax.set_ylabel(ylabel)
    ax.set_xlim(gamma_s.min(), gamma_s.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if filename_prefix == "equilibrium_env_vs_gamma_s":
        ax.set_ylim(0, 1)

    fig.savefig(
        savedir / f"{filename_prefix}_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_timeseries_env_by_rationality(
    mean_environment: npt.NDArray[np.float64],
    rationality: npt.NDArray[np.float64],
    quality_label: str,
    savedir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2), constrained_layout=True)

    repeats, _, steps = mean_environment.shape
    t = np.arange(steps)

    for rationality_idx, lambda_value in enumerate(rationality):
        mean = mean_environment[:, rationality_idx].mean(axis=0)
        std = mean_environment[:, rationality_idx].std(axis=0, ddof=1)
        ci = 1.97 * std / np.sqrt(repeats)

        ax.plot(t, mean, label=f"$\\lambda = {lambda_value:.2f}$")
        ax.fill_between(t, mean - ci, mean + ci, alpha=0.3)

    ax.set_ylabel("Mean environment")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(
        savedir / f"time_series_mean_env_by_rationality_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def _run_heatmap(args: argparse.Namespace) -> None:
    quality_label = "low" if args.quick else "high"

    peer_pressure_label = "peerconst" if args.const_peer_pressure else "peerrandomised"
    dynamic_action_preference = "adaptive" if args.dynamic_action_preference else None
    adaptive_label = dynamic_action_preference or "nonadaptive"
    predict_label = "predictive" if args.neighborhood_prediction else "nonpredictive"

    if args.metric in ("environment", "both"):
        if adaptive_label == "adaptive":
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

        mode_label = "" if adaptive_label == "adaptive" else f"{adaptive_label}_"
        data = np.load(DATA_DIR / f"{filename}.npz")

        plot_heatmap(
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
        plot_heatmap(
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
        plot_heatmap(
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

    if args.metric in ("fourier", "both"):
        filename = (
            f"fourier_power_rationality_vs_gamma_s"
            f"_{adaptive_label}_{predict_label}_{peer_pressure_label}"
            f"_{quality_label}_quality"
        )
        data = np.load(DATA_DIR / f"{filename}.npz")

        plot_heatmap(
            savedir=FIGURES_DIR,
            values=data["power"],
            rationality=data["rationality"],
            gamma_s=data["gamma_s"],
            filename=f"{filename}.pdf",
            cbar_label="Power at dominant frequency",
            vmin=0,
            vmax=30000,
        )


def _run_curve(args: argparse.Namespace) -> None:
    quality_label = "low" if args.quick else "high"

    if args.metric == "eq-env-vs-rationality":
        data = np.load(DATA_DIR / f"eq_env_vs_rationality_{quality_label}_quality.npz")
        plot_eq_env_vs_rationality(
            mean_environment=data["mean_environment"],
            rationality=data["rationality"],
            savedir=FIGURES_DIR,
            quality_label=quality_label,
        )
        return

    peer_pressure_label = "peerconst" if args.const_peer_pressure else "peerrandomised"

    if args.metric == "eq-env-vs-gamma":
        data = np.load(
            DATA_DIR
            / f"eq_env_vs_gamma_s_{peer_pressure_label}_{quality_label}_quality.npz"
        )
        plot_curve_vs_gamma(
            values=data["mean_environment"],
            gamma_s=data["gamma_s"],
            ylabel=r"Equilibrium mean environment ($\overline{n}^*$)",
            filename_prefix="equilibrium_env_vs_gamma_s",
            savedir=FIGURES_DIR,
            quality_label=f"{peer_pressure_label}_{quality_label}",
        )
        return

    if args.metric == "fourier-vs-gamma":
        data = np.load(
            DATA_DIR
            / (
                f"fourier_power_vs_gamma_s_{peer_pressure_label}_"
                f"{quality_label}_quality.npz"
            )
        )
        plot_curve_vs_gamma(
            values=data["power"],
            gamma_s=data["gamma_s"],
            ylabel="Dominant Fourier power",
            filename_prefix="fourier_power_vs_gamma_s",
            savedir=FIGURES_DIR,
            quality_label=f"{peer_pressure_label}_{quality_label}",
        )
        return

    raise ValueError(f"Unsupported curve metric: {args.metric}")


def _run_timeseries(args: argparse.Namespace) -> None:
    if args.metric != "env-vs-rationality":
        raise ValueError(f"Unsupported time-series metric: {args.metric}")

    quality_label = "low" if args.quick else "high"
    data = np.load(
        DATA_DIR
        / f"time_series_mean_env_for_varying_rationality_{quality_label}_quality.npz"
    )

    plot_timeseries_env_by_rationality(
        mean_environment=data["mean_environment"],
        rationality=data["rationality"],
        quality_label=quality_label,
        savedir=FIGURES_DIR,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified plotting runner for heatmap sweeps, 1D curves, and time-series "
            "experiments."
        )
    )
    framework_parsers = parser.add_subparsers(dest="framework", required=True)

    heatmap = framework_parsers.add_parser("heatmap")
    heatmap.add_argument(
        "--metric",
        choices=["environment", "fourier", "both"],
        default="both",
    )
    heatmap.add_argument("--quick", action="store_true")
    heatmap.add_argument("--const-peer-pressure", action="store_true")
    heatmap.add_argument(
        "--dynamic-action-preference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    heatmap.add_argument("--neighborhood-prediction", default=None)
    heatmap.add_argument("--variance-slice-lambda", type=float)
    heatmap.add_argument("--autocorr-slice-lambda", type=float)
    heatmap.add_argument("--acf-lambda", type=float)
    heatmap.add_argument("--acf-gamma", type=float)

    curve = framework_parsers.add_parser("curve")
    curve.add_argument(
        "--metric",
        choices=["eq-env-vs-rationality", "eq-env-vs-gamma", "fourier-vs-gamma"],
        required=True,
    )
    curve.add_argument("--quick", action="store_true")
    curve.add_argument("--const-peer-pressure", action="store_true")

    timeseries = framework_parsers.add_parser("timeseries")
    timeseries.add_argument("--metric", choices=["env-vs-rationality"], required=True)
    timeseries.add_argument("--quick", action="store_true")

    return parser


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.framework == "heatmap":
        _run_heatmap(args)
        return

    if args.framework == "curve":
        _run_curve(args)
        return

    if args.framework == "timeseries":
        _run_timeseries(args)
        return

    parser.error(f"Unsupported framework: {args.framework}")


if __name__ == "__main__":
    run()

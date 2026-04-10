import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from abm_project.mean_field import solve as solve_mean_field
from abm_project.plotting import configure_mpl
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def get_parallel_workers(default_workers: int = 2) -> int:
    """Return a conservative process count; override with ABM_MAX_WORKERS."""
    cpu_count = os.cpu_count() or 1
    env_value = os.getenv("ABM_MAX_WORKERS")
    if env_value is not None:
        try:
            return max(1, min(int(env_value), cpu_count))
        except ValueError:
            print(
                "Invalid ABM_MAX_WORKERS="
                f"'{env_value}'. Falling back to "
                f"{default_workers}."
            )
    return max(1, min(default_workers, cpu_count))


def get_chunksize(n_tasks: int, workers: int, default: int = 1) -> int:
    """Get executor map chunksize; override with ABM_CHUNKSIZE."""
    env_value = os.getenv("ABM_CHUNKSIZE")
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            print(f"Invalid ABM_CHUNKSIZE='{env_value}'. Falling back to auto.")
    if n_tasks <= 0:
        return 1
    return max(1, min(default, n_tasks // max(1, workers * 8)))


def set_fixed_initial_conditions(
    model: VectorisedModel,
    rng: np.random.Generator,
    fixed_n0: float | None,
    fixed_m0: float | None,
):
    """Override initial simulation state when fixed means are requested."""
    if fixed_n0 is None and fixed_m0 is None:
        return

    history_len = min(model.memory_count, model.action.shape[0])

    if fixed_n0 is not None:
        model.environment[:history_len] = fixed_n0
        model.curr_s = np.zeros(model.num_agents, dtype=np.float64) + 0.5
        for _ in range(100):
            model.adapt(model.environment[0])

    if fixed_m0 is not None:
        p_target = (fixed_m0 + 1) / 2
        n_coop = int(round(p_target * model.num_agents))
        base_actions = np.full(model.num_agents, -1, dtype=np.int64)
        base_actions[:n_coop] = 1

        # Shuffle each row so the global action mean is controlled without spatial bias.
        for t in range(history_len):
            model.action[t] = rng.permutation(base_actions)


def run_single_comparison(task_args):
    (
        repeat_idx,
        lambda_idx,
        rationality,
        width,
        height,
        steps,
        memory_count,
        gamma_s,
        env_gamma,
        recovery,
        pollution,
        neighb_prediction_option,
        radius_option,
        const_peer_pressure,
        b2_value,
        fixed_n0,
        fixed_m0,
    ) = task_args

    num_agents = width * height
    rng = np.random.default_rng(repeat_idx)
    env_update_fn = piecewise_exponential_update(
        recovery=recovery,
        pollution=pollution,
        gamma=env_gamma,
    )

    b_1 = np.full(num_agents, 1.0)
    b_2 = np.full(num_agents, b2_value) if const_peer_pressure else None

    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=rationality,
        neighb_prediction_option=neighb_prediction_option,
        severity_benefit_option="adaptive",
        radius_option=radius_option,
        max_storage=steps,
        b_1=b_1,
        b_2=b_2,
        gamma_s=gamma_s,
    )

    set_fixed_initial_conditions(
        model=model,
        rng=rng,
        fixed_n0=fixed_n0,
        fixed_m0=fixed_m0,
    )

    n0 = float(model.environment[0].mean())
    m0 = float(model.action[0].mean())
    b_effective = float(model.b[0].mean())
    c_effective = float(model.b[1].mean())

    model.run(steps)
    sim_env_timeseries = model.environment[: model.time + 1].mean(axis=1)

    _, (mean_field_env, _, _, _, _) = solve_mean_field(
        b=b_effective,
        c=c_effective,
        alpha=model.alpha,
        beta=model.beta,
        pollution=pollution,
        recovery=recovery,
        n_update_rate=env_gamma,
        s_update_rate=gamma_s,
        n0=n0,
        m0=m0,
        num_steps=steps,
        rationality=rationality,
    )

    return (repeat_idx, lambda_idx, sim_env_timeseries, mean_field_env)


def save_comparison_plot(
    t: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    sim_env: npt.NDArray[np.float64],
    mean_field_env: npt.NDArray[np.float64],
    quality_label: str,
    savepath: Path,
):
    configure_mpl()

    n_lambdas = len(lambdas)
    n_cols = 2 if n_lambdas > 1 else 1
    n_rows = math.ceil(n_lambdas / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.0 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    repeats = sim_env.shape[0]
    for idx, (axis, lmbda) in enumerate(zip(axes, lambdas, strict=False)):
        sim_mean = sim_env[:, idx].mean(axis=0)
        if repeats > 1:
            sim_std = sim_env[:, idx].std(axis=0, ddof=1)
            ci = 1.97 * sim_std / np.sqrt(repeats)
        else:
            ci = np.zeros_like(sim_mean)

        mf_mean = mean_field_env[:, idx].mean(axis=0)

        axis.plot(t, sim_mean, color="tab:blue", label="Simulation")
        axis.fill_between(t, sim_mean - ci, sim_mean + ci, color="tab:blue", alpha=0.2)
        axis.plot(t, mf_mean, color="tab:orange", linestyle="--", label="Mean field")

        axis.set_title(f"$\\lambda={lmbda:.2f}$")
        axis.set_ylim(0, 1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes[n_lambdas:]:
        axis.set_visible(False)

    for axis in axes[:n_lambdas]:
        axis.set_xlabel("Time step")
    for axis in axes[::n_cols]:
        axis.set_ylabel("Mean environment")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Mean-field vs averaged simulation time series ({quality_label} quality)",
        y=1.02,
    )

    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def main(
    data_dir: Path,
    figures_dir: Path,
    quality_label: str,
    repeats: int,
    lambdas: npt.NDArray[np.float64],
    width: int,
    height: int,
    steps: int,
    memory_count: int,
    gamma_s: float,
    env_gamma: float,
    recovery: float,
    pollution: float,
    neighb_prediction_option: str | None,
    radius_option: str,
    const_peer_pressure: bool,
    b2_value: float,
    fixed_n0: float | None,
    fixed_m0: float | None,
):
    n_tasks = repeats * len(lambdas)
    sim_env = np.empty((repeats, len(lambdas), steps + 1), dtype=np.float64)
    mean_field_env = np.empty((repeats, len(lambdas), steps + 1), dtype=np.float64)

    tasks = [
        (
            repeat_idx,
            lambda_idx,
            lmbda,
            width,
            height,
            steps,
            memory_count,
            gamma_s,
            env_gamma,
            recovery,
            pollution,
            neighb_prediction_option,
            radius_option,
            const_peer_pressure,
            b2_value,
            fixed_n0,
            fixed_m0,
        )
        for repeat_idx in range(repeats)
        for lambda_idx, lmbda in enumerate(lambdas)
    ]

    workers = get_parallel_workers()
    chunksize = get_chunksize(n_tasks=n_tasks, workers=workers)
    print(f"Running {n_tasks} simulations with {workers} worker process(es).")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for repeat_idx, lambda_idx, sim_series, mf_series in tqdm(
            executor.map(run_single_comparison, tasks, chunksize=chunksize),
            total=n_tasks,
            desc="Running simulation and mean-field comparisons",
        ):
            sim_env[repeat_idx, lambda_idx] = sim_series
            mean_field_env[repeat_idx, lambda_idx] = mf_series

    t = np.arange(steps + 1, dtype=np.float64)

    data_dir.mkdir(parents=True, exist_ok=True)
    data_out = (
        data_dir / f"mean_field_vs_simulated_timeseries_{quality_label}_quality.npz"
    )
    np.savez(
        data_out,
        t=t,
        lambdas=lambdas,
        simulated_mean_environment=sim_env,
        mean_field_environment=mean_field_env,
        repeats=repeats,
        width=width,
        height=height,
        steps=steps,
        memory_count=memory_count,
        gamma_s=gamma_s,
        env_gamma=env_gamma,
        recovery=recovery,
        pollution=pollution,
        const_peer_pressure=const_peer_pressure,
        b2_value=b2_value,
        radius_option=radius_option,
        fixed_n0=np.nan if fixed_n0 is None else fixed_n0,
        fixed_m0=np.nan if fixed_m0 is None else fixed_m0,
        neighborhood_prediction=(
            "none" if neighb_prediction_option is None else neighb_prediction_option
        ),
    )

    figure_out = (
        figures_dir / f"mean_field_vs_simulated_timeseries_{quality_label}_quality.pdf"
    )
    save_comparison_plot(
        t=t,
        lambdas=lambdas,
        sim_env=sim_env,
        mean_field_env=mean_field_env,
        quality_label=quality_label,
        savepath=figure_out,
    )

    print(f"Saved data to {data_out}")
    print(f"Saved figure to {figure_out}")


if __name__ == "__main__":
    DATA_DIR = Path("data")
    FIGURES_DIR = Path("results/figures")

    DEFAULT_LAMBDAS = [1.0, 3.0, 4.3, 5.5, 6.5]

    QUICK_REPEATS = 2
    QUICK_STEPS = 400

    FULL_REPEATS = 25
    FULL_STEPS = 1500

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--memory-count", type=int, default=10)
    parser.add_argument("--gamma-s", type=float, default=0.015)
    parser.add_argument("--env-gamma", type=float, default=0.01)
    parser.add_argument("--recovery", type=float, default=1.0)
    parser.add_argument("--pollution", type=float, default=1.0)
    parser.add_argument(
        "--neighborhood-prediction",
        choices=["none", "linear", "logistic", "true_pref"],
        default="none",
    )
    parser.add_argument(
        "--radius-option",
        choices=["single", "all"],
        default="single",
        help="Neighborhood radius mode. Use 'all' for a more mean-field-like coupling.",
    )
    parser.add_argument(
        "--const-peer-pressure",
        action="store_true",
        help=(
            "Use homogeneous b_2 for all agents. Recommended for mean-field comparison."
        ),
    )
    parser.add_argument("--b2-value", type=float, default=1.0)
    parser.add_argument(
        "--fixed-n0",
        type=float,
        help="Optional fixed initial mean environment in [0, 1].",
    )
    parser.add_argument(
        "--fixed-m0",
        type=float,
        help="Optional fixed initial mean action in [-1, 1].",
    )
    args = parser.parse_args()

    if args.quick:
        quality_label = "low"
        repeats = QUICK_REPEATS
        steps = QUICK_STEPS
    else:
        quality_label = "high"
        repeats = FULL_REPEATS
        steps = FULL_STEPS

    if args.repeats is not None:
        if args.repeats < 1:
            parser.error("--repeats must be >= 1")
        repeats = args.repeats

    if args.steps is not None:
        if args.steps < 1:
            parser.error("--steps must be >= 1")
        steps = args.steps

    if len(args.lambdas) == 0:
        parser.error("--lambdas must contain at least one value")

    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be >= 1")

    if args.memory_count < 1:
        parser.error("--memory-count must be >= 1")

    if args.gamma_s <= 0:
        parser.error("--gamma-s must be > 0")

    if args.env_gamma <= 0:
        parser.error("--env-gamma must be > 0")

    if args.recovery <= 0 or args.pollution <= 0:
        parser.error("--recovery and --pollution must be > 0")

    if args.b2_value <= 0:
        parser.error("--b2-value must be > 0")

    if args.fixed_n0 is not None and not (0 <= args.fixed_n0 <= 1):
        parser.error("--fixed-n0 must be in [0, 1]")

    if args.fixed_m0 is not None and not (-1 <= args.fixed_m0 <= 1):
        parser.error("--fixed-m0 must be in [-1, 1]")

    neighb_prediction_option = (
        None if args.neighborhood_prediction == "none" else args.neighborhood_prediction
    )

    main(
        data_dir=DATA_DIR,
        figures_dir=FIGURES_DIR,
        quality_label=quality_label,
        repeats=repeats,
        lambdas=np.array(args.lambdas, dtype=np.float64),
        width=args.width,
        height=args.height,
        steps=steps,
        memory_count=args.memory_count,
        gamma_s=args.gamma_s,
        env_gamma=args.env_gamma,
        recovery=args.recovery,
        pollution=args.pollution,
        neighb_prediction_option=neighb_prediction_option,
        radius_option=args.radius_option,
        const_peer_pressure=args.const_peer_pressure,
        b2_value=args.b2_value,
        fixed_n0=args.fixed_n0,
        fixed_m0=args.fixed_m0,
    )

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ruptures import Pelt
from tqdm import tqdm

from abm_project.cluster_analysis import cluster_time_series
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel


def test_cluster_across_memory(option):
    # Parameters for the simulation
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 500
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None
    rationality = 1.8
    simmer_time = 1
    neighb_prediction_option = "linear"
    # severity_benefit_option = None
    memory_values = [x for x in range(5, 101, 5)]

    # Analysis parameters
    replicates = 20
    critical_times = {m: [] for m in memory_values}
    cluster_n = {n: [] for n in memory_values}
    largest_cluster = {n: [] for n in memory_values}

    for mem_count in memory_values:
        for rep in tqdm(
            range(replicates), desc=f"Memory Count: {mem_count}, Replicate: "
        ):
            print(f"Running replicate {rep + 1} for memory count {mem_count}...")
            results_dir = Path("cluster_analysis_results")
            results_dir.mkdir(exist_ok=True)

            memory_count = mem_count

            # start = time.time()
            model = VectorisedModel(
                num_agents=num_agents,
                width=width,
                height=height,
                memory_count=memory_count,
                rng=rng,
                env_update_fn=env_update_fn,
                rationality=rationality,
                simmer_time=simmer_time,
                neighb_prediction_option=neighb_prediction_option,
                severity_benefit_option=None,
                max_storage=num_steps,
            )
            model.run(num_steps)
            # end = time.time()

            Nc, C1 = cluster_time_series(model=model, option=option)

            # 3) Detect transition time t_c via change‐point on Nc(t)
            algo = Pelt(model="rbf", min_size=5).fit(Nc)
            bkpts = algo.predict(pen=3)  # list of breakpoints
            critical_times[mem_count].append(bkpts[0])  # first change point

    # 4) Summarize: average t_c vs memory
    for m in memory_values:
        print(f"Memory {m}:\nmean t_c = {np.mean(critical_times[m]):.1f}")
        print(f"std = {np.std(critical_times[m]):.1f}")

    return critical_times, cluster_n, largest_cluster, memory_values


# TODO: Fix it. Doesn't work yet
def plot_cluster_across_memory(critical_times, savedir):
    savedir = savedir or Path(".")

    memory_values = list(critical_times.keys())

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        memory_values,
        [np.mean(critical_times[m]) for m in memory_values],
        yerr=[np.std(critical_times[m]) for m in memory_values],
        fmt="o-",
        capsize=5,
    )
    plt.xlabel("Memory Count")
    plt.ylabel("Critical Time (t_c)")
    plt.title("Critical Time vs Memory Count")
    plt.grid()

    if savedir:
        plt.savefig(
            savedir / f"cluster_across_memory_{option}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved as 'cluster_across_memory_{option}.png'")
    plt.show()


def plot_ncluster_given_memory(model, option, savedir):
    savedir = savedir or Path(".")

    # Nc, C1 = cluster_time_series(model=model, option=option)
    nc, c1 = cluster_time_series(model=model, option=option)
    print(f"Number of clusters at each time step: {nc}")
    print(f"Largest cluster fraction at each time step: {c1 / model.num_agents}")

    plt.figure(figsize=(10, 6))
    plt.plot(nc, label="Number of Clusters")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Clusters")
    plt.title("Number of Clusters Over Time")
    plt.legend()
    plt.grid()

    if savedir:
        plt.savefig(
            savedir
            / f"n_clusters_given_{model.memory_count}_{option}_\
            {model.severity_benefit_option}_{model.rationality}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Plot saved as 'n_clusters_given_{model.memory_count}_\
            {option}_{model.severity_benefit_option}_{model.rationality}.png'"
        )
    plt.show()


def plot_ncluster_across_memory(cluster_n, savedir):
    savedir = savedir or Path(".")

    memory_values = list(cluster_n.keys())

    fig, ax = plt.figure(figsize=(10, 6))
    for mem in memory_values:
        ax.plot(cluster_n[mem], label=f"Memory {mem}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Number of Clusters Over Time for Different Memory Counts")
    ax.legend()
    ax.grid()
    if savedir:
        fig.savefig(
            savedir / f"n_clusters_across_memory_{option}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'n_clusters_across_memory.png'")
    plt.show()


# def main():
# Set model parameters


if __name__ == "__main__":
    # option  = "action"
    option = "environment"
    # critical_times, cluster_n, largest_cluster, _ = test_cluster_across_memory(option)
    # plot_cluster_across_memory(critical_times,
    #    savedir=Path("plots/cluster_analysis_results"))
    # for m in cluster_n.keys():
    #     print(cluster_n[m])

    # plot_ncluster_across_memory(cluster_n, savedir=Path("cluster_analysis_results"))
    num_agents = 2500
    width = 50
    height = 50
    num_steps = 500
    memory_count = 50
    env_update_fn = piecewise_exponential_update(alpha=1, beta=1, rate=0.01)
    rng = None

    start = time.time()
    model = VectorisedModel(
        num_agents=num_agents,
        width=width,
        height=height,
        memory_count=memory_count,
        rng=rng,
        env_update_fn=env_update_fn,
        rationality=10,
        simmer_time=1,
        neighb_prediction_option="linear",
        severity_benefit_option="adaptive",
        max_storage=num_steps,
    )
    model.run(num_steps)
    end = time.time()

    print(f"Simulation completed in {end - start:.2f} seconds.")

    plot_ncluster_given_memory(
        model=model, option=option, savedir=Path("cluster_analysis_results")
    )
    print("Cluster analysis completed and plot generated.")

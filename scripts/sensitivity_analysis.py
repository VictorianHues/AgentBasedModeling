import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
from SALib.sample import saltelli
import tqdm as tqdm
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as sobol_plot
from SALib.analyze.sobol import to_df

from abm_project.plotting import (
    get_data_directory
)
from abm_project.utils import piecewise_exponential_update
from abm_project.vectorised_model import VectorisedModel

#get parameter values 
def problem():
    problem = {
        "num_vars" : 3,
        "names": ["width","rationality","memory_count"], 
        "bounds": [[5,50], [0,20], [1,10]]
    }
    return problem

def sample_parameter_space():
    problem = {
        "num_vars" : 3,
        "names": ["width","rationality","memory_count"], 
        "bounds": [[5,50], [0,20], [1,10]]
    }
    param_values = sobol_sample.sample(problem, 64)
    return param_values

def run_single_parameter_set(width, rationality, memory_count):
    num_runs = 15  # number of runs for the batch simulation
    steps = 1000  # number of simulation steps

    kwargs = {
        "num_agents": width*width,
        "width": width,
        "height": width,
        "memory_count": memory_count,
        "env_update_fn": piecewise_exponential_update(alpha=1, beta=1, rate=0.01),
        "rng": None,
        "rationality": rationality,
        "max_storage": steps,
        "moore": True,
        "simmer_time": 1,
        "neighb_prediction_option": "linear",
        "severity_benefit_option": None,
        "prop_pessimistic": 1.0,
        "pessimism_level": 1.0,
    }

    env_means = np.zeros((num_runs, 1))
    #env_vars = np.zeros((steps + 1,))
    action_means = np.zeros((num_runs, 1))
    #action_vars = np.zeros((steps + 1,))

    for run in range(num_runs):
        model = VectorisedModel(**kwargs)
        model.run(steps)

        final_environment = model.environment[-1,:]
        final_action = model.action[-1,:]
        env_means[run] = np.mean(final_environment)
        action_means[run] = np.mean(final_action)

        del model

    data_file_path = get_data_directory("output_results.npz")
    np.savez(
        data_file_path,
        env_means=env_means,
        action_means=action_means
    )
    mean_env = np.mean(env_means)
    mean_action = np.mean(action_means)

    return mean_env, mean_action

def gather_output_statistics():

    param_values = sample_parameter_space()
    environment_output = []
    action_output = []

    for i, (width, rationality, memory_count) in enumerate(tqdm.tqdm(param_values)):
        width = int(width)
        memory_count = int(memory_count)
        mean_env, mean_action = run_single_parameter_set(width, rationality, memory_count)
        environment_output.append(mean_env)
        action_output.append(mean_action)
        np.savez("data/output_environment.npz", env=environment_output, act=action_output)
    return environment_output, action_output
    

def plotting_output():
    load_data = np.load("data/output_environment.npz")
    environment_output = load_data["env"]
    action_output = load_data["act"]
    
    plt.hist(environment_output, bins=30)
    plt.title("Environment output")
    plt.show()

    plt.hist(action_output, bins=30)
    plt.title("Action output")
    plt.show()

def sobol_sensitivity():
    load_data = np.load("data/output_environment.npz")
    environment_output = load_data["env"]
    action_output = load_data["act"]

    problem_dict = problem()

    sobol_environment = sobol.analyze(problem_dict, environment_output)
    sobol_action = sobol.analyze(problem_dict, action_output)

    return sobol_environment, sobol_action
    

if __name__ == "__main__":
    gather_output_statistics()
    #plotting_output()
    #sobol_sensitivity()
    
import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
experiment_directory = os.path.join(root, "experiments", "linear")

sys.path.append(root)

import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as npe
import experiments.linear.linear_models
import concurrent.futures

with open(os.path.join(experiment_directory, "parameters_linear.json"), "r") as f:
    params = json.load(f)

## Simulation Code ##

# General function which runs the simulations for all model types
def nonstationary_test_experiment(f, filename):

    tau = params["tau"]
    t = np.linspace(0, 1, params["length"])

    print(f"Running {filename} experiment")

    for _ in range(int(params["N_replicates"])):

        evidence, significance_level = nt.nonstationarity_test((f(), t, tau), 
                            theta_range=params["theta_range"],
                            delta_range=params["delta_range"], E_range=params["E_range"],
                            lambda1=params["lambda1"], lambda2=params["lambda2"],
                            p=params["p"], resolution=params["resolution"])
        
        theta, delta = npe.compute_posterior_weighted_parameters((f(), t, tau), 
                            theta_range=params["theta_range"],
                            delta_range=params["delta_range"], E_range=params["E_range"],
                            lambda1=params["lambda1"], lambda2=params["lambda2"],
                            p=params["p"], resolution=params["resolution"])
        
        out_path = os.path.join(f"{filename}.csv")
        with open(out_path, 'a') as file:
            file.write(f"{evidence},{significance_level},{theta},{delta}\n")
        # results.append([evidence, significance_level, theta, delta])

    # np.savetxt(f"{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f,%0.4f", header = "evidence, significance_level, theta, delta")

## Run ##

def run_single_experiment(experiment):
    dynamic_function_name = "generate_" + experiment["name"]
    dynamic_function = getattr(experiments.linear.linear_models, dynamic_function_name)
    filename = os.path.join(experiment_directory, experiment["name"])
    nonstationary_test_experiment(dynamic_function, filename)
    return experiment["name"]

if __name__ == "__main__":
    # Parallel execution of experiments
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_experiment, experiment): experiment for experiment in params["experiments"]}
        for future in concurrent.futures.as_completed(futures):
            experiment_name = futures[future]["name"]
            try:
                result = future.result()
                print(f"Finished {result} experiment")
            except Exception as exc:
                print(f"Experiment {experiment_name} generated an exception: {exc}")

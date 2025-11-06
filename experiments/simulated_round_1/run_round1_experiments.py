import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Use __file__ instead of __name__
experiment_directory = os.path.join(root, "experiments", "simulated_round_1")

sys.path.append(root)

import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as nse
import experiments.simulated_round_1.models
import csv

with open(os.path.join(experiment_directory, "parameters_round1.json"), "r") as f:
    params = json.load(f)

def write_to_file(filename, row):

    csv_path = os.path.join(experiment_directory, f"{filename}.csv")

    # Open file in append mode for results
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(row)

## Simulation Code ##

def write_to_file(filename, row):
    csv_path = os.path.join(experiment_directory, f"{filename}.csv")
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(row)

# Pass experiment_params as an argument
def nonstationary_test_experiment(f, filename, experiment_params):
    tau = params["tau"]
    resolution = params["resolution"]
    t = np.linspace(0, 1, experiment_params["time_series_length"])
    N_replicates = int(params["N_replicates"])

    results = []

    for _ in range(N_replicates):
        system = f()

        evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
            (system, t, tau),
            theta_range=experiment_params["theta_range"],
            delta_range=experiment_params["delta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            p=params["p"],
            resolution=resolution
        )

        evidence_linear, significance_level_linear, bayes_factor_error_linear = nt.nonstationarity_test_linear(
            (system, t, tau),
            delta_range=experiment_params["delta_range"],
            E_range=params["E_range"],
            lambda2=params["lambda2"],
            p=params["p"],
            resolution=resolution
        )

        posterior_weighted_theta, posterior_weighted_delta = nse.compute_posterior_weighted_parameters(
            (system, t, tau),
            theta_range=experiment_params["theta_range"],
            delta_range=experiment_params["delta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            p=params["p"],
            resolution=resolution
        )

        row = [evidence, significance_level, bayes_factor_error,
               evidence_linear, significance_level_linear, bayes_factor_error_linear,
               posterior_weighted_theta, posterior_weighted_delta]

        write_to_file(filename, row)

        """
        results.append(np.array([evidence, significance_level, bayes_factor_error,
                                 evidence_linear, significance_level_linear, bayes_factor_error_linear,
                                 posterior_weighted_theta, posterior_weighted_delta]))
        """

    # np.savetxt(f"{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,", header = "evidence, significance_level, bayes factor error, linear evidence, linear significance_level, linear bayes factor error, posterior weighted theta, posterior weighted delta", delimiter=",")

## Run ##

if __name__ == "__main__":
    # Run the simulation for the stationary model

    for experiment in params["experiments"][4:]:  # Limiting to two experiments for testing
        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.simulated_round_1.models, dynamic_function_name)

        dir_name = experiment_directory + "/" + experiment["name"]

        print(f"Starting {experiment['name']} experiment")
        nonstationary_test_experiment(dynamic_function, dir_name, experiment["parameters"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + dir_name + ".csv")
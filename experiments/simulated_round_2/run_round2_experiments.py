import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
experiment_directory = os.path.join(root, "experiments", "simulated_round_2/")

sys.path.append(root)

import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import experiments.simulated_round_2.nonlinear_models
import csv

with open(os.path.join(experiment_directory, "parameters_round2.json"), "r") as f:
    params = json.load(f)

## Simulation Code ##

def write_to_file(filename, row):

    csv_path = os.path.join(experiment_directory, f"{filename}.csv")

    # Open file in append mode for results
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(row)

# General function which runs the simulations for all model types
def nonstationary_test_experiment(f, experiment_params, filename):

    tau = params["tau"]
    N_replicates = params["N_replicates"]
    resolution = params["resolution"]

    default_length = experiment_params["default_length"]
    default_observation_noise = experiment_params["default_obs_noise"]
    default_process_noise = experiment_params["default_process_noise"]
    nonstat_param_base = experiment_params["nonstat_param_base"]

    print(f"Running nonstationarity test for {filename} with parameters:")
    print(f"Default Length: {default_length}, Default Observation Noise: {default_observation_noise}, Default Process Noise: {default_process_noise}, Nonstationarity Parameter Base: {nonstat_param_base}")
    print("Time Series Lengths:", experiment_params["time_series_length"])

    header = ["nonstationary level", "time series length", "observation noise", "process noise", "evidence", "significance_level", "bayes factor error"]
    write_to_file(filename, header)

    for rep in range(N_replicates):
        print(f"Replication {rep + 1}/{N_replicates}")
        # Time Series Length
        for length in experiment_params["time_series_length"]:
            t = np.linspace(0, 1, length)
            for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
                time_series = f(length, nonstat_param_base, nonstat_slope, default_observation_noise, default_process_noise)
                evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                    (time_series, t, tau),
                    theta_range=params["theta_range"],
                    delta_range=params["delta_range"],
                    E_range=params["E_range"],
                    lambda1=params["lambda1"],
                    lambda2=params["lambda2"],
                    p=params["p"],
                    resolution=resolution
                )
                row = [i, length, default_observation_noise, default_process_noise, evidence, significance_level, bayes_factor_error]
                write_to_file(filename, row)

        print("Observation Noise:", experiment_params["obs_noise"])
        # Observation Noise
        for observation_noise in experiment_params["obs_noise"]:
            t = np.linspace(0, 1, default_length)
            for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
                time_series = f(default_length, nonstat_param_base, nonstat_slope, observation_noise, default_process_noise)
                evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                    (time_series, t, tau),
                    theta_range=params["theta_range"],
                    delta_range=params["delta_range"],
                    E_range=params["E_range"],
                    lambda1=params["lambda1"],
                    lambda2=params["lambda2"],
                    p=params["p"],
                    resolution=resolution
                )
                row = [i, default_length, observation_noise, default_process_noise, evidence, significance_level, bayes_factor_error]
                write_to_file(filename, row)

        print("Process Noise:", experiment_params["process_noise"])
        # Process Noise
        for process_noise in experiment_params["process_noise"]:
            t = np.linspace(0, 1, default_length)
            for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
                time_series = f(default_length, nonstat_param_base, nonstat_slope, default_observation_noise, process_noise)
                evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                    (time_series, t, tau),
                    theta_range=params["theta_range"],
                    delta_range=params["delta_range"],
                    E_range=params["E_range"],
                    lambda1=params["lambda1"],
                    lambda2=params["lambda2"],
                    p=params["p"],
                    resolution=resolution
                )
                row = [i, default_length, default_observation_noise, process_noise, evidence, significance_level, bayes_factor_error]
                write_to_file(filename, row)

if __name__ == "__main__":
    # Run the simulation for the stationary model

    for experiment in params["experiments"]:
        if experiment["name"] == "food_chain":
            print("Skipping Food Chain Experiment")
            continue

        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.simulated_round_2.nonlinear_models, dynamic_function_name)

        nonstationary_test_experiment(dynamic_function, experiment["parameters"], experiment_directory + experiment["name"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + experiment_directory + experiment["name"] + ".csv")
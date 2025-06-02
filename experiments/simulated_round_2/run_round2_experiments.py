import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
experiment_directory = os.path.join(root, "experiments", "simulated_round_2")

sys.path.append(root)

import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import experiments.simulated_round_2.nonlinear_models

with open(os.path.join(experiment_directory, "parameters_round2.json"), "r") as f:
    params = json.load(f)

## Simulation Code ##

# General function which runs the simulations for all model types
def nonstationary_test_experiment(f, experiment_params, filename):

    tau = params["tau"]

    results = []

    default_length = experiment_params["time_series_length"][1]
    default_observation_noise = experiment_params["obs_noise"][1]
    default_process_noise = experiment_params["process_noise"][1]

    # Time Series Length
    for length in experiment_params["time_series_length"]:

        t = np.linspace(0, 1, length)

        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            # Generate the time series with the specified nonstationarity parameters
            time_series = f(length, experiment_params["nonstat_param_base"], nonstat_slope, default_observation_noise, default_process_noise)

            evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"]
            )

            results.append(np.array([i, length, default_observation_noise, default_process_noise, evidence, significance_level, bayes_factor_error]))

    # Observation Noise
    for observation_noise in experiment_params["obs_noise"]:

        t = np.linspace(0, 1, default_length)

        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            # Generate the time series with the specified nonstationarity parameters
            time_series = f(default_length, experiment_params["nonstat_param_base"], nonstat_slope, observation_noise, default_process_noise)

            evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"]
            )

            results.append(np.array([i, default_length, observation_noise, default_process_noise, evidence, significance_level, bayes_factor_error]))

    # Process Noise
    for process_noise in experiment_params["process_noise"]:

        t = np.linspace(0, 1, default_length)

        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            # Generate the time series with the specified nonstationarity parameters
            time_series = f(default_length, experiment_params["nonstat_param_base"], nonstat_slope, default_observation_noise, process_noise)

            evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"]
            )

            results.append(np.array([i, default_length, default_observation_noise, process_noise, evidence, significance_level, bayes_factor_error]))


    print(results)
    results = np.vstack(results)
    print(results)

    np.savetxt(f"{filename}.csv", results, fmt="%i, %i, %0.1f, %0.2f, %0.4f,%0.4f,%0.4f", header = "nonstationary level, time series length, observation noise, process noise, evidence, significance_level, bayes factor error")

## Run ##

if __name__ == "__main__":
    # Run the simulation for the stationary model

    for experiment in params["experiments"]:
        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.simulated_round_2.nonlinear_models, dynamic_function_name)

        nonstationary_test_experiment(dynamic_function, experiment["parameters"], experiment_directory + experiment["name"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + experiment_directory + experiment["name"] + ".csv")
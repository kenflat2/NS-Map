import os
import sys
import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import experiments.simulated_round_2.nonlinear_models
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__file__))
# experiment_directory = os.path.join(root, "experiments", "simulated_round_2/")
sys.path.append(root)

with open(os.path.join(root, "parameters_round2_calibrated_process_noise2.json"), "r") as f:
    params = json.load(f)

## Simulation Code ##

def write_to_file(filename, rows):
    csv_path = os.path.join(root, f"{filename}_calibrated_process_noise2.csv")
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerows(rows)

# General function which runs the simulations for all model types
def run_replication(rep, f, experiment_params, params):
    tau = params["tau"]
    resolution = params["resolution"]
    default_length = experiment_params["default_length"]
    default_observation_noise = experiment_params["default_obs_noise"]
    default_process_noise = experiment_params["default_process_noise"]
    nonstat_param_base = experiment_params["nonstat_param_base"]
    rows = []
    # Time Series Length
    for length in experiment_params["time_series_length"]:
        t = np.linspace(0, 1, length)
        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            time_series = f(length, nonstat_param_base, nonstat_slope, default_observation_noise, default_process_noise)
            evidence, significance_level = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=resolution
            )
            row = [i, length, default_observation_noise, default_process_noise, evidence, significance_level]
            rows.append(row)
    # Observation Noise
    for observation_noise in experiment_params["obs_noise"]:
        t = np.linspace(0, 1, default_length)
        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            time_series = f(default_length, nonstat_param_base, nonstat_slope, observation_noise, default_process_noise)
            evidence, significance_level = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=resolution
            )
            row = [i, default_length, observation_noise, default_process_noise, evidence, significance_level]
            rows.append(row)
    # Process Noise
    for process_noise in experiment_params["process_noise"]:
        t = np.linspace(0, 1, default_length)
        for i, nonstat_slope in enumerate(experiment_params["nonstat_param_slope"]):
            time_series = f(default_length, nonstat_param_base, nonstat_slope, default_observation_noise, process_noise)
            evidence, significance_level = nt.nonstationarity_test(
                (time_series, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=resolution
            )
            row = [i, default_length, default_observation_noise, process_noise, evidence, significance_level]
            rows.append(row)
    return rows

def nonstationary_test_experiment_parallel(f, experiment_params, filename):
    N_replicates = params["N_replicates"]
    print(f"Running nonstationarity test for {filename} with parameters:")
    print(f"Default Length: {experiment_params['default_length']}, Default Observation Noise: {experiment_params['default_obs_noise']}, Default Process Noise: {experiment_params['default_process_noise']}, Nonstationarity Parameter Base: {experiment_params['nonstat_param_base']}")
    print("Time Series Lengths:", experiment_params["time_series_length"])
    header = ["nonstationary level", "time series length", "observation noise", "process noise", "evidence", "significance_level"]
    write_to_file(filename, [header])
    all_rows = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_replication, rep, f, experiment_params, params) for rep in range(N_replicates)]
        for i, future in enumerate(as_completed(futures)):
            rows = future.result()
            all_rows.extend(rows)
            print(f"Replication {i + 1}/{N_replicates} finished.")
    write_to_file(filename, all_rows)

if __name__ == "__main__":
    # Run the simulation for the stationary model
    for experiment in params["experiments"]:
        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.simulated_round_2.nonlinear_models, dynamic_function_name)
        nonstationary_test_experiment_parallel(dynamic_function, experiment["parameters"], experiment["name"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + os.path.join(root, experiment["name"] + ".csv"))
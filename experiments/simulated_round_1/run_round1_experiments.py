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
from concurrent.futures import ProcessPoolExecutor
import csv

with open(os.path.join(experiment_directory, "parameters_round1.json"), "r") as f:
    params = json.load(f)

def write_to_file(filename, row):

    csv_path = os.path.join(experiment_directory, f"{filename}.csv")

    # Write header if file does not exist or is empty
    header = [
               "evidence", "significance_level",
               "evidence_DLM", "significance_level_DLM",
               "posterior_weighted_theta", "posterior_weighted_delta",
               "prediction_skill_ns", "prediction_skill_DLM"
    ]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

## Simulation Code ##

"""
def write_to_file(filename, row):
    csv_path = os.path.join(experiment_directory, f"{filename}.csv")
    with open(csv_path, "a") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(row)
"""

# Pass experiment_params as an argument
def nonstationary_test_experiment(f, filename, experiment_params):
    tau = params["tau"]
    resolution = params["resolution"]
    t = np.linspace(0, 1, experiment_params["time_series_length"])
    N_replicates = int(params["N_replicates"])

    for _ in range(N_replicates):
        system = f()

        evidence, significance_level, best_params_ns = nt.nonstationarity_test(
            (system, t, tau),
            theta_range=experiment_params["theta_range"],
            delta_range=experiment_params["delta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            p=params["p"],
            resolution=resolution,
            return_best_params=True
        )

        best_theta_ns, best_delta_ns, best_E_ns = best_params_ns

        evidence_DLM, significance_level_DLM, best_params_DLM = nt.nonstationarity_test_linear(
            (system, t, tau),
            delta_range=experiment_params["delta_range"],
            E_range=params["E_range"],
            lambda2=params["lambda2"],
            p=params["p"],
            resolution=resolution,
            return_best_params=True
        )

        best_delta_DLM, best_E_DLM = best_params_DLM

        # Compute prediction skill (R^2) for best parameters
        Xr = system
        Xemb_DLM, Y, tx = ns.delayEmbed(Xr, best_E_DLM, tau, t=t)
        Yhat_DLM = ns.leaveOneOut(Xemb_DLM, Y, tx, 0, best_delta_DLM)
        ss_res_DLM = np.sum((Y.flatten() - Yhat_DLM.flatten())**2)
        ss_tot_DLM = np.sum((Y.flatten() - np.mean(Y))**2)
        prediction_skill_DLM = 1 - ss_res_DLM / ss_tot_DLM if ss_tot_DLM > 0 else np.nan
        
        Xemb_ns, Y, tx = ns.delayEmbed(Xr, best_E_ns, tau, t=t)
        Yhat_ns = ns.leaveOneOut(Xemb_ns, Y, tx, best_theta_ns, best_delta_ns)
        ss_res_ns = np.sum((Y.flatten() - Yhat_ns.flatten())**2)
        ss_tot_ns = np.sum((Y.flatten() - np.mean(Y))**2)
        prediction_skill_ns = 1 - ss_res_ns / ss_tot_ns if ss_tot_ns > 0 else np.nan

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

        row = [evidence, significance_level,
               evidence_DLM, significance_level_DLM,
               posterior_weighted_theta, posterior_weighted_delta,
               prediction_skill_ns, prediction_skill_DLM]

        write_to_file(filename, row)

        """
        results.append(np.array([evidence, significance_level, bayes_factor_error,
                                 evidence_DLM, significance_level_DLM, bayes_factor_error_DLM,
                                 posterior_weighted_theta, posterior_weighted_delta]))
        """

    # np.savetxt(f"{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,", header = "evidence, significance_level, bayes factor error, linear evidence, linear significance_level, linear bayes factor error, posterior weighted theta, posterior weighted delta", delimiter=",")

## Run ##

def process_experiment(experiment):
    dynamic_function_name = "generate_" + experiment["name"]
    dynamic_function = getattr(experiments.simulated_round_1.models, dynamic_function_name)
    dir_name = experiment_directory + "/" + experiment["name"]
    print(f"Starting {experiment['name']} experiment")
    nonstationary_test_experiment(dynamic_function, dir_name, experiment["parameters"])
    print(f"Finished {experiment['name']} experiment")
    print("Results saved to " + dir_name + ".csv")

def run_experiments_parallel():
    # You can adjust the slice below to control which experiments to run
    experiments_to_run = params["experiments"][4:]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_experiment, experiment) for experiment in experiments_to_run]
        for future in futures:
            future.result()

if __name__ == "__main__":
    run_experiments_parallel()
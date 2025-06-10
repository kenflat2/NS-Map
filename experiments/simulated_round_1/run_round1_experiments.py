import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
experiment_directory = os.path.join(root, "experiments", "simulated_round_1")

sys.path.append(root)

import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as nse
import experiments.simulated_round_1.models


with open(os.path.join(experiment_directory, "parameters_nonlinear.json"), "r") as f:
    params = json.load(f)

## Simulation Code ##

# General function which runs the simulations for all model types
def nonstationary_test_experiment(f, filename):

    # Compute the posterior weighted parameters for the given system,
    # then compute the normal nonstationary test then the linear nonstationary test

    tau = params["tau"]
    t = np.linspace(0, 1, params["length"])
    N_replicates = int(params["N_replicates"])

    results = []

    for _ in range(N_replicates):
        system = f()

        evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
            (system, t, tau),
            theta_range=params["theta_range"],
            delta_range=params["delta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            p=params["p"]
        )

        evidence_linear, significance_level_linear, bayes_factor_error_linear = nt.nonstationarity_test_linear(
            (system, t, tau),
            theta_range=params["theta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            p=params["p"]
        )

        posterior_weighted_theta, posterior_weighted_delta = nse.compute_posterior_weighted_parameters(
            (system, t, tau),
            theta_range=params["theta_range"],
            delta_range=params["delta_range"],
            E_range=params["E_range"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            p=params["p"]
        )

        results.append(np.array([evidence, significance_level, bayes_factor_error,
                                 evidence_linear, significance_level_linear, bayes_factor_error_linear,
                                 posterior_weighted_theta, posterior_weighted_delta]))

    np.savetxt(f"{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,", header = "evidence, significance_level, bayes factor error, linear evidence, linear significance_level, linear bayes factor error, posterior weighted theta, posterior weighted delta", delimiter=",")

## Run ##

if __name__ == "__main__":
    # Run the simulation for the stationary model

    for experiment in params["experiments"]:
        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.linear.linear_models, dynamic_function_name)

        nonstationary_test_experiment(dynamic_function, experiment_directory + experiment["name"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + experiment_directory + experiment["name"] + ".csv")
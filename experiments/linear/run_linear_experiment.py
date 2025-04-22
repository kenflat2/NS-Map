import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt
import experiments.linear.linear_models

experiment_directory = "experiments/linear/"

with open(experiment_directory + "parameters_linear.json", "r") as f:
    params = json.load(f)

## Simulation Code ##

# General function which runs the simulations for all model types
def nonstationary_test_experiment(f, filename):

    tau = params["tau"]
    t = np.linspace(0, 1, params["length"])

    results = np.array([nt.nonstationarity_test((f(), t, tau), 
                        theta_range=params["theta_range"],
                        delta_range=params["delta_range"], E_range=params["E_range"],
                        lambda1=params["lambda1"], lambda2=params["lambda2"],
                        p=params["p"])
                        for i in range(int(params["N_replicates"]))])
    
    np.savetxt(f"{experiment_directory}{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f", header = "evidence, significance_level, bayes factor error")

## Run ##

if __name__ == "__main__":
    # Run the simulation for the stationary model

    for experiment in params["experiments"]:
        dynamic_function_name = "generate_" + experiment["name"]
        dynamic_function = getattr(experiments.linear.linear_models, dynamic_function_name)

        nonstationary_test_experiment(dynamic_function, experiment_directory + experiment["name"])
        print(f"Finished {experiment['name']} experiment")
        print("Results saved to " + experiment_directory + experiment["name"] + ".csv")
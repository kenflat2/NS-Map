import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt

with open("parameters.json", "r") as f:
    params = json.load(f)

## Simulation Code ##

# General function which runs the simulations for all model types
def runSimulation(f, filename):

    results = np.array([nt.nonstationarity_test(f(), theta_range=params["theta_range"],
                        delta_range=params["delta_range"], E_max=params["E_max"],
                        lambda1=params["lambda1"], lambda2=params["lambda2"],
                        p=params["p"])
                        for i in range(int(params["N_replicates"]))])
    
    np.savetxt(f"{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f", header = "evidence, significance_level, bayes factor error")

## Run ##

if __name__ == "__main__":
    # Run the simulation for the stationary model
    runSimulation(ns.generate_stationary_data, "stationary_linear")
    
    # Run the simulation for the nonstationary model
    runSimulation(ns.generate_nonstationary_data, "nonstationary_linear")
    
    # Run the simulation for the nonstationary model with a different parameter set
    runSimulation(ns.generate_nonstationary_data2, "nonstationary_linear2")
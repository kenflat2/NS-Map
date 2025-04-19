import json
import numpy as np
import src.NSMap as ns
import src.NonstationarityTest as nt

with open("parameters.json", "r") as f:
    params = json.load(f)

# General function which runs the simulations for all model types
def runSimulation(f, filename, N_replicates = 100, E_max = 8):
    results = np.array([ns.get_delta_agg(f(), E_max, return_forecast_skill=True)
                        for i in range(int(N_replicates))])
    np.savetxt(f"../results/linear_results/{filename}.csv", results, fmt="%0.4f,%0.4f,%0.4f", header = "delta, theta, r squared")
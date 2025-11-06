import numpy as np
import pandas as pd
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as nse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# Load parameters from JSON file
def load_params(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Standardize function (assumes mean/std normalization)
def standardize(ts):
    return (ts - np.mean(ts)) / np.std(ts)

# Main experiment runner
def process_species(i, species, table, params, lengths, maxLen, output_dir):
    ts = standardize(table[:,i])
    tally = 0
    tau = params["tau"]
    for length in lengths:
        for start in np.arange(0, maxLen - length+1, step=params["year_steps"]):
            ts_chunk = ts[start:length+start]
            t = np.linspace(0, 1, length)
            evidence, significance_level, bayes_factor_error = nt.nonstationarity_test(
                (ts_chunk, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=params["resolution"]
            )
            posterior_weighted_theta, posterior_weighted_delta = nse.compute_posterior_weighted_parameters(
                (ts_chunk, t, tau),
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=params["resolution"]
            )
            out_path = os.path.join(output_dir, f"{species}_time_window.csv")
            with open(out_path, 'a') as f:
                f.write(f"{int(length)},{int(start)},{evidence},{significance_level},{bayes_factor_error},{posterior_weighted_theta},{posterior_weighted_delta}\n")
            tally += 1
            print(f"{species}: {tally}")

def run_experiment():
    params = load_params("experiments/baltic_sea_mesocosm/baltic_sea_mesocosm_params.json")
    filename = params["data_csv"]
    year_steps = params["year_steps"]
    output_dir = params["output_dir"]
    df = pd.read_csv(filename, encoding="utf-8", na_filter=False)
    table = df.to_numpy()
    maxLen = table[:,0].shape[0]
    lengths = np.arange(year_steps*4, maxLen+1, step=year_steps)
    species_list = list(df.columns[1:].to_numpy())

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_species, i+1, species, table, params, lengths, maxLen, output_dir)
                    for i, species in enumerate(species_list)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    print("Running Baltic Sea Mesocosm")
    run_experiment()
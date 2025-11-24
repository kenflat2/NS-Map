import numpy as np
import pandas as pd
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as nse
import src.NSMap as ns
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# Load parameters from JSON file
def load_params(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Main experiment runner
def process_species(i, species, table, params, lengths, maxLen, output_dir):

    ts = ns.standardize(table[:,i])    

    # length, starting index, delta, theta, r_sqrd
    # results = np.zeros((n_rows, 5))
    tally = 0
    tau = params["tau"]

    for length in lengths:
        for start in np.arange(0, maxLen - length+1, step=12):
            ts_chunk = ns.standardize(ts[start:length+start])
            t = np.linspace(0, 1, length)
            data = (ts_chunk, t, tau)

            evidence, significance_level = nt.nonstationarity_test(
                data,
                theta_range=params["theta_range"],
                delta_range=params["delta_range"],
                E_range=params["E_range"],
                lambda1=params["lambda1"],
                lambda2=params["lambda2"],
                p=params["p"],
                resolution=params["resolution"]
            )

            posterior_weighted_theta, posterior_weighted_delta = nse.compute_posterior_weighted_parameters(
                data,
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
                f.write(f"{int(length)},{int(start)},{evidence},{significance_level},{posterior_weighted_theta},{posterior_weighted_delta}\n")
            tally += 1
            print(f"{species}: {tally}")

def run_experiment():
    params = load_params("experiments/newport_line/newport_params.json")
    filename = params["data_csv"]
    # year_steps = params["year_steps"]
    output_dir = params["output_dir"]

    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(filename, encoding="utf-8", dtype=np.float32)
    table = df.to_numpy(dtype=np.float32)

    maxLen = table.shape[0]
    lengths = np.arange(24,maxLen+1,step=12)
    species_list = list(df.columns.to_numpy())

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_species, i, species, table, params, lengths, maxLen, output_dir)
                    for i, species in enumerate(species_list)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    print("Running Newport Line Experiment")
    run_experiment()
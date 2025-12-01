import numpy as np
import pandas as pd
import src.NonstationarityTest as nt
import src.NSHyperparameterEstimation as nse
import src.NSMap as ns
import json
import os
from concurrent.futures import ProcessPoolExecutor

def load_params(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_species_series(species, df, df_trapping):
    from datetime import date
    total_months_elapsed = 46 * 12 + 1
    month_counter = 7
    year_counter = 1977
    table = df[["month", "day", "year", "period", "species"]].query("period >= 0")
    table_species = table.query(f"species == '{species}'")[["month", "year"]].to_numpy(dtype=np.int64)
    trapping_days = df_trapping.query("sampled == 1")[["month", "year"]].to_numpy()
    ts = np.zeros(total_months_elapsed) * np.nan
    ts_i = 0
    while not (month_counter > 7 and year_counter > 2023):
        did_sampling_occur = np.any(np.logical_and(trapping_days[:,0] == month_counter, trapping_days[:,1] == year_counter))
        if did_sampling_occur:
            is_sample_valid = np.logical_and(table_species[:,0] == month_counter, table_species[:,1] == year_counter)
            ts[ts_i] = np.sum(table_species[is_sample_valid])
        ts_i += 1
        month_counter += 1
        if month_counter > 12:
            month_counter = 1
            year_counter += 1
    t = np.linspace(0,1, num= total_months_elapsed)
    return (ts, t)

def process_species(species, params, df, df_trapping, lengths, maxLen, output_dir):
    ts, t = get_species_series(species, df, df_trapping)
    ts = ns.standardize(ts)
    tau = params["tau"]
    tally = 0
    out_path = os.path.join(output_dir, f"{species}_time_window.csv")
    # Write header if file does not exist
    if not os.path.exists(out_path):
        with open(out_path, 'w') as f:
            f.write("length,start,evidence,significance_level,posterior_weighted_theta,posterior_weighted_delta\n")
    for length in lengths:
        for start in np.arange(0, maxLen - length+1, step=12):
            ts_chunk = ns.standardize(ts[start:length+start])
            t_chunk = np.linspace(0, 1, length)
            data = (ts_chunk, t_chunk, tau)
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
            with open(out_path, 'a') as f:
                f.write(f"{int(length)},{int(start)},{evidence},{significance_level},{posterior_weighted_theta},{posterior_weighted_delta}\n")
            tally += 1
            print(f"{species}: {tally}")

def run_experiment():
    params = load_params("experiments/rodent/portal_params.json")
    filename = "experiments/rodent/Portal_rodent.csv"
    trapping_filename = "experiments/rodent/Portal_rodent_trapping.csv"
    output_dir = params["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.read_csv(filename, encoding="utf-8", na_filter=False)
    df_trapping = pd.read_csv(trapping_filename, encoding="utf-8", na_filter=False)
    # Get unique species
    species_list = df["species"].dropna().unique()
    maxLen = 46 * 12 + 1
    year_steps = 24
    lengths = np.arange(year_steps*4,maxLen+1,step=year_steps)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_species, species, params, df, df_trapping, lengths, maxLen, output_dir)
                    for species in species_list]
        for future in futures:
            future.result()

if __name__ == "__main__":
    print("Running Portal Rodent Experiment")
    run_experiment()
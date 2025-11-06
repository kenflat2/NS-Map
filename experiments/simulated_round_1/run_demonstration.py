import sys
import os
from pathlib import Path
root = Path().resolve().parent.parent
sys.path.append(str(root))  # Add the root of the project to the local path

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from experiments.simulated_round_1.models import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.NonstationarityTest import marginalize_likelihood_1d, marginalize_likelihood_2d
from src.NonstationarityTest import prior_E
import pickle

if __name__ == "__main__":

    names = ["Stationary\nLinear", "Nonstationary\nLinear", "Stationary\nNonlinear", "Nonstationary\nNonlinear"]

    stuff_dict = {}

    linear = generate_linear()
    linear_nonstat = generate_linear_nonstat()
    logistic = generate_logistic()
    logistic_nonstat = generate_logistic_nonstat()
    food_chain = generate_food_chain()
    food_chain_nonstat = generate_food_chain_nonstat()

    # Parameters
    with open(os.path.join(experiment_directory, "parameters_round1.json"), "r") as f:
        params = json.load(f)

    # Example data
    t = np.linspace(0, 1, 200)
    tau = 1
    data = (linear, t, tau)

    E_range = params["E_range"]
    embedding_dimensions = np.arange(E_range[0], E_range[1])

    lambda1 = params["lambda1"]
    lambda2 = params["lambda2"]
    p = params["p"]

    # Run marginalization for all four example datasets
    for (name, experiment, series) in zip(names, params["experiments"][:4], [linear, linear_nonstat, logistic, logistic_nonstat]):

        theta_range = experiment["parameters"]["theta_range"]
        delta_range = experiment["parameters"]["delta_range"]

        print(f"{name}: {linear.shape}, {linear_nonstat.shape}, {logistic.shape}, {logistic_nonstat.shape}")

        data = (series, t, tau)

        embedding_dimensions = np.arange(E_range[0], E_range[1])

        marginal_likelihood_s = np.zeros(E_range[1] - E_range[0])
        marginal_likelihood_ns = np.zeros(E_range[1] - E_range[0])
        marginal_error_s = np.zeros(E_range[1] - E_range[0])
        marginal_error_ns = np.zeros(E_range[1] - E_range[0])

        print(marginal_likelihood_s.shape, marginal_likelihood_ns.shape)

        for E in embedding_dimensions:
            data_E = (data[0], data[1], E, data[2])

            # Marginalize the likelihood for SMap (null)
            marginal_likelihood_s[E], marginal_error_s[E] = marginalize_likelihood_1d(data_E, theta_range, lambda1)

            # Marginalize the likelihood for NSMap
            marginal_likelihood_ns[E], marginal_error_ns[E] = marginalize_likelihood_2d(data_E, theta_range, delta_range, lambda1, lambda2)

        stuff_dict[name] = (series, t, marginal_likelihood_s, marginal_likelihood_ns)
    
    with open("stuff_dict.pkl", "wb") as f:
        pickle.dump(stuff_dict, f)
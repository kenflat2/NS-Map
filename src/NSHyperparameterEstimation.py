import src.NSMap as ns
import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad
from src.NonstationarityTest import prior_E, posterior_2d

def compute_posterior_weighted_parameters(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5):

    numerator1 = 0
    numerator2 = 0
    Z = 0

    for E in range(E_range[0], E_range[1] + 1):

        data_E = (data[0], data[1], E, data[2])

        # posterior_integrands, theta_values, delta_values = compute_posterior_on_grid(data_E, theta_range, delta_range, lambda1, lambda2)
        # posterior_weights = posterior_integrands / np.sum(posterior_integrands)

        # Integrate the posterior times theta
        numerator1 += posterior_weighted_function_grid(lambda theta, delta: theta, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

        # Integrate the posterior times delta
        numerator2 += posterior_weighted_function_grid(lambda theta, delta: delta, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

        Z += posterior_weighted_function_grid(lambda theta, delta: 1, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

    return numerator1 / Z, numerator2 / Z

def posterior_weighted_function(func, data, param1_range, param2_range, lambda1=1.0, lambda2=1.0):
    post_weighted_func = lambda theta, delta, data, lambda1, lambda2: posterior_2d(theta, delta, data, lambda1, lambda2) * func(theta, delta)

    integral, error = dblquad(
        post_weighted_func,
        param1_range[0], param1_range[1],  # Integration limits for theta
        lambda theta: param2_range[0], lambda theta: param2_range[1],  # Integration limits for delta
        args=(data, lambda1, lambda2),
        epsrel=1e-5, epsabs=1e-5
    )

    return integral

def compute_posterior_on_grid(data, param1_range, param2_range, lambda1=1.0, lambda2=1.0, resolution=10):
    theta_values = np.linspace(param1_range[0], param1_range[1], resolution)
    delta_values = np.linspace(param2_range[0], param2_range[1], resolution)

    theta, delta = np.meshgrid(theta_values, delta_values)

    posterior_grid_evals = np.nan_to_num(posterior_2d(theta_values, delta_values, data, lambda1, lambda2))

    dtheta = (param1_range[1] - param1_range[0]) / (resolution - 1)
    ddelta = (param2_range[1] - param2_range[0]) / (resolution - 1)

    posterior_grid_volumes = posterior_grid_evals * dtheta * ddelta

    return posterior_grid_volumes, theta_values, delta_values

def posterior_weighted_function_grid(func, data, param1_range, param2_range, lambda1=1.0, lambda2=1.0, resolution=10):
    theta_values = np.linspace(param1_range[0], param1_range[1], resolution)
    delta_values = np.linspace(param2_range[0], param2_range[1], resolution)

    theta, delta = np.meshgrid(theta_values, delta_values, indexing='ij')

    post_weighted_func = np.nan_to_num(posterior_2d(theta_values, delta_values, data, lambda1, lambda2) * func(theta, delta), 0)

    dtheta = (param1_range[1] - param1_range[0]) / (resolution - 1)
    ddelta = (param2_range[1] - param2_range[0]) / (resolution - 1)

    integral = np.sum(post_weighted_func) * dtheta * ddelta

    return integral
import src.NSMap as ns
import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad
from src.NonstationarityTest import prior_E, posterior_2d

def compute_posterior_weighted_parameters(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5):

    numerator1 = 0
    numerator2 = 0
    Z = 0

    for E in E_range:

        data_E = (data[0], data[1], E, data[2])

        # Integrate the posterior times theta
        numerator1 += posterior_weighted_function(lambda theta, delta: theta, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

        # Integrate the posterior times delta
        numerator2 += posterior_weighted_function(lambda theta, delta: delta, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

        Z += posterior_weighted_function(lambda theta, delta: 1, data_E, theta_range, delta_range, lambda1, lambda2) * prior_E(E, p)

    return numerator1 / Z, numerator2 / Z

def posterior_weighted_function(func, data, param1_range, param2_range, lambda1=1.0, lambda2=1.0):
    post_weighted_func = lambda theta, delta : posterior_2d(theta, delta) * func(theta, delta)
	
    integral, error = dblquad(
        post_weighted_func,
        param1_range[0], param1_range[1],  # Integration limits for theta
        lambda theta: param2_range[0], lambda theta: param2_range[1],  # Integration limits for delta
        args=(data, lambda1, lambda2),
        epsrel=1e-5, epsabs=1e-5
    )

    return integral
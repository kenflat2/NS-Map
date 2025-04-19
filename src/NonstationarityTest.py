import src.NSMap as ns
import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad

# Define the likelihood function
def likelihood(data, theta, delta):
    (Xr, tx, E, tau) = data

    Xemb, Y, tx = ns.delayEmbed(Xr, E, tau, t=tx)

    # Replace this with the actual likelihood function
    # This is just a placeholder example
    return np.exp(ns.logLikelihood(Xemb, Y, tx, theta, delta))

# Define the exponentially decreasing prior function
def prior_1d(theta, lambda_t=1.0):
    return lambda_t * np.exp(-lambda_t * theta)

# Define the exponentially decreasing prior function
def prior_2d(theta, delta, lambda_d = 1.0, lambda_t=1.0):
    return lambda_t *lambda_d * np.exp(-lambda_t * theta) * np.exp(-lambda_d * delta)

def prior_E(E, p=0.5):
    return ((1 - p) ** E) * p

# Define the posterior function (likelihood * prior)
def posterior_1d(param1, data, lambda1=1.0):
    return likelihood(data, param1, 0) * prior_1d(param1, lambda1)

# Define the posterior function (likelihood * prior)
def posterior_2d(param1, param2, data, lambda1=1.0, lambda2=1.0):
    return likelihood(data, param1, param2) * prior_2d(param1, param2, lambda1, lambda2)

# Function to marginalize the posterior over the parameter space
def marginalize_likelihood_1d(data, param_range, lambda_=1.0):
    integral, error = quad(
        posterior_1d,
        param_range[0], param_range[1],  # Integration limits for the parameter
        args=(data, lambda_),
        epsrel=1e-5, epsabs=1e-5
    )
    return integral, error

# Function to marginalize the posterior over the parameter space
def marginalize_likelihood_2d(data, param1_range, param2_range, lambda1=1.0, lambda2=1.0):
    integral, error = dblquad(
        posterior_2d,
        param1_range[0], param1_range[1],  # Integration limits for param1
        lambda param1: param2_range[0], lambda param1: param2_range[1],  # Integration limits for param2
        args=(data, lambda1, lambda2),
        epsrel=1e-5, epsabs=1e-5
    )
    return integral, error

def marginalize_likelihood_E(marginal_likelihood_func, data, param_range, lambda_=1.0, p=0.5):
    marginal_likelihood = 0
    marginal_error = 0
    for E in range(param_range[0], param_range[1]):
        integral, error = marginal_likelihood_func(data, E, lambda_, p) * prior_E(E)
        marginal_likelihood += integral
        marginal_error += error
    return marginal_likelihood, marginal_error

def compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5):

    marginal_likelihood_s = 0
    marginal_error_s = 0
    marginal_likelihood_ns = 0
    marginal_error_ns = 0

    for E in range(E_range[0], E_range[1]):
        data_E = (data[0], data[1], E, data[2])

        # Marginalize the likelihood for SMap (null)
        integral_s, error_s = marginalize_likelihood_1d(data_E, theta_range, lambda1)
        marginal_likelihood_s += integral_s * prior_E(E)
        marginal_error_s += error_s

        # Marginalize the likelihood for NSMap
        integral_ns, error_ns = marginalize_likelihood_2d(data_E, theta_range, delta_range, lambda1, lambda2)
        marginal_likelihood_ns += integral_ns * prior_E(E)
        marginal_error_ns += error_ns

    # Compute the Bayes Factor
    bayes_factor = marginal_likelihood_ns / marginal_likelihood_s
    error_bf = marginal_error_s / marginal_likelihood_s + marginal_error_ns / marginal_likelihood_ns
    return bayes_factor, error_bf

# Function to perform the nonstationarity test
# Inputs: 
#   - data: the time series data Xr
#   - theta_range: range for theta parameter (tuple)
#   - delta_range: range for delta parameter (tuple)
#   - E_range: range for E parameter (tuple)
#   - lambda1: parameter for the prior on theta (float)
#   - lambda2: parameter for the prior on delta (float)
#   - p: parameter for the prior on E (float)
# Outputs:
#   - log_bayes_factor: log of the Bayes Factor between stationary and nonstationary model (float)
#   - significance_level: significance level of the test (float)
#   - error_bf: error estimate for the Bayes Factor (float)
def nonstationarity_test(data, theta_range=(0.0, 4.0), delta_range=(0.0, 4.0), E_range=(0, 8), lambda1=1.0, lambda2=1.0, p=0.5):
    # Compute the Bayes Factor
    bayes_factor, error_bf = compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1, lambda2, p)

    # Compute the log Bayes Factor
    log_bayes_factor = np.log(bayes_factor)

    # Compute the significance level
    significance_level = 1 - np.exp(-log_bayes_factor)

    return log_bayes_factor, significance_level, error_bf
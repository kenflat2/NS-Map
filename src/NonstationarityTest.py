import src.NSMap as ns
import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad

# Define a global constant for numerical integration tolerance
INTEGRATION_TOL = 1e-2

# Define the likelihood function
def likelihood(data, theta, delta):
    (Xr, tx, E, tau) = data

    Xemb, Y, tx = ns.delayEmbed(Xr, E, tau, t=tx)

    # Rescale the hyperparameters so their range is greater
    # theta = np.exp(theta) - 1
    # delta = np.exp(delta) - 1

    return np.exp(ns.logLikelihood(Xemb, Y, tx, theta, delta))

# Define the exponentially decreasing prior function
def prior_1d(theta, lambda_t=1.0):
    return lambda_t * np.exp(-lambda_t * theta)

# Define the exponentially decreasing prior function
def prior_2d(theta, delta, lambda_d = 1.0, lambda_t=1.0):
    return lambda_t *lambda_d * np.exp(-lambda_t * theta) * np.exp(-lambda_d * delta)

def prior_E(E, p=0.5):
    return ((1 - p) ** E) * p

"""
# Define the posterior function (likelihood * prior)
def posterior_1d(param1, data, lambda1=1.0):
    return likelihood(data, param1, 0) * prior_1d(param1, lambda1)
"""

def posterior_1d_linear(param1, data, lambda1=1.0):
    return likelihood(data, 0, 0) * prior_1d(param1, lambda1)

"""
# Define the posterior function (likelihood * prior)
def posterior_2d(param1, param2, data, lambda1=1.0, lambda2=1.0):
    return likelihood(data, param1, param2) * prior_2d(param1, param2, lambda1, lambda2)
"""

def posterior_1d(param1, data, lambda1=1.0):
    # Vectorize the computation
    vectorized_likelihood = np.vectorize(lambda p: likelihood(data, p, 0))
    vectorized_prior = np.vectorize(lambda p: prior_1d(p, lambda1))
    return vectorized_likelihood(param1) * vectorized_prior(param1)

def posterior_2d(param1, param2, data, lambda1=1.0, lambda2=1.0):
    """
    Compute the posterior for 2D parameters in a vectorized manner.
    
    Args:
        param1: Array of parameter values for theta.
        param2: Array of parameter values for delta.
        data: Data tuple (Xr, tx, E, tau).
        lambda1: Prior parameter for theta.
        lambda2: Prior parameter for delta.

    Returns:
        Array of posterior values for each (param1, param2) pair.
    """
    # Ensure param1 and param2 are NumPy arrays
    param1 = np.atleast_1d(param1)
    param2 = np.atleast_1d(param2)
    
    # Create a grid of param1 and param2 values
    param1_grid, param2_grid = np.meshgrid(param1, param2, indexing='ij')
    
    # Flatten the grids for vectorized computation
    param1_flat = param1_grid.ravel()
    param2_flat = param2_grid.ravel()
    
    # Compute likelihood and prior for all (param1, param2) pairs
    likelihood_vals = np.array([likelihood(data, p1, p2) for p1, p2 in zip(param1_flat, param2_flat)])
    prior_vals = prior_2d(param1_flat, param2_flat, lambda1, lambda2)
    
    # Reshape the results back to the grid shape
    posterior_vals = likelihood_vals * prior_vals
    return posterior_vals.reshape(param1_grid.shape)

# Function to marginalize the posterior over the parameter space
def marginalize_likelihood_1d(data, param_range, lambda_=1.0):
    integral, error = quad(
        posterior_1d,
        param_range[0], param_range[1],  # Integration limits for the parameter
        args=(data, lambda_),
        epsrel=INTEGRATION_TOL, epsabs=INTEGRATION_TOL
    )
    return integral, error

# Function to marginalize the posterior over the parameter space
def marginalize_likelihood_2d(data, param1_range, param2_range, lambda1=1.0, lambda2=1.0):
    integral, error = dblquad(
        posterior_2d,
        param1_range[0], param1_range[1],  # Integration limits for param1
        lambda param1: param2_range[0], lambda param1: param2_range[1],  # Integration limits for param2
        args=(data, lambda1, lambda2),
        epsrel=INTEGRATION_TOL, epsabs=INTEGRATION_TOL
    )
    return integral, error

def marginalize_likelihood_1d_trapezoidal(data, param_range, lambda_=1.0, resolution=10, integrand=posterior_1d):
    """
    Marginalize the posterior over the parameter space using the trapezoidal rule (1D).

    Args:
        data: Data tuple.
        param_range: Tuple specifying the range of the parameter (min, max).
        lambda_: Prior parameter.
        resolution: Number of points for the trapezoidal rule.

    Returns:
        Integral value.
    """
    param_vals = np.linspace(param_range[0], param_range[1], resolution)
    posterior_vals = integrand(param_vals, data, lambda_)
    integral = np.trapz(posterior_vals, x=param_vals)

    return integral, None

def marginalize_likelihood_2d_trapezoidal(data, param1_range, param2_range, lambda1=1.0, lambda2=1.0, resolution=10):
    """
    Marginalize the posterior over the parameter space using the trapezoidal rule (2D).

    Args:
        data: Data tuple.
        param1_range: Tuple specifying the range of the first parameter (min, max).
        param2_range: Tuple specifying the range of the second parameter (min, max).
        lambda1: Prior parameter for the first parameter.
        lambda2: Prior parameter for the second parameter.
        resolution: Number of points along each axis for the trapezoidal rule.

    Returns:
        Integral value.
    """
    param1_vals = np.linspace(param1_range[0], param1_range[1], resolution)
    param2_vals = np.linspace(param2_range[0], param2_range[1], resolution)

    posterior_vals = np.nan_to_num(posterior_2d(param1_vals, param2_vals, data, lambda1, lambda2), nan=0.0)
    integral = np.trapz(np.trapz(posterior_vals, x=param2_vals, axis=1), x=param1_vals)

    return integral, None

def marginalize_likelihood_E(marginal_likelihood_func, data, param_range, lambda_=1.0, p=0.5):
    marginal_likelihood = 0
    marginal_error = 0
    for E in range(param_range[0], param_range[1]):
        integral, error = marginal_likelihood_func(data, E, lambda_, p) * prior_E(E)
        marginal_likelihood += integral
        marginal_error += error
    return marginal_likelihood, marginal_error

def compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5, debug=False):

    marginal_likelihood_s = 0
    marginal_error_s = 0
    marginal_likelihood_ns = 0
    marginal_error_ns = 0

    embedding_dimensions = np.arange(E_range[0], E_range[1])

    marginal_likelihood_s = np.zeros(E_range[1] - E_range[0])
    marginal_likelihood_ns = np.zeros(E_range[1] - E_range[0])
    marginal_error_s = np.zeros(E_range[1] - E_range[0])
    marginal_error_ns = np.zeros(E_range[1] - E_range[0])

    for i, E in enumerate(embedding_dimensions):
        data_E = (data[0], data[1], E, data[2])

        # Marginalize the likelihood for SMap (null)
        marginal_likelihood_s[i], marginal_error_s[i] = marginalize_likelihood_1d_trapezoidal(data_E, theta_range, lambda1)

        # Marginalize the likelihood for NSMap
        # marginal_likelihood_ns[E], marginal_error_ns[E] = marginalize_likelihood_2d(data_E, theta_range, delta_range, lambda1, lambda2)
        marginal_likelihood_ns[i], marginal_error_ns[i] = marginalize_likelihood_2d_trapezoidal(data_E, theta_range, delta_range, lambda1, lambda2)

    marginal_likelihood_s = np.dot(marginal_likelihood_s, prior_E(embedding_dimensions))
    marginal_error_s = np.sum(marginal_error_s)
    marginal_likelihood_ns = np.dot(marginal_likelihood_ns, prior_E(embedding_dimensions))
    marginal_error_ns = np.sum(marginal_error_ns)

    # Compute the Bayes Factor
    bayes_factor = marginal_likelihood_ns / marginal_likelihood_s
    error_bf = marginal_error_s / marginal_likelihood_s + marginal_error_ns / marginal_likelihood_ns
    if debug:
        return bayes_factor, error_bf, marginal_likelihood_s, marginal_error_s, marginal_likelihood_ns, marginal_error_ns
    else:
        # Return the Bayes Factor and error estimate
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
    evidence = 10 * np.log10(bayes_factor)

    # Compute the significance level
    significance_level = 1 - (10 ** (-evidence/10))

    return evidence, significance_level, error_bf

def compute_bayes_factor_linear(data, delta_range=(0.0, 4.0), E_range=(0, 8), lambda2=1.0, p=0.5, debug=False):
    # Compute the Bayes Factor for the linear case
    embedding_dimensions = np.arange(E_range[0], E_range[1])

    marginal_likelihood_s = np.zeros(E_range[1] - E_range[0])
    marginal_likelihood_ns = np.zeros(E_range[1] - E_range[0])
    marginal_error_s = np.zeros(E_range[1] - E_range[0])
    marginal_error_ns = np.zeros(E_range[1] - E_range[0])

    for idx, E in enumerate(embedding_dimensions):
        data_E = (data[0], data[1], E, data[2])

        # Stationary: theta=0, delta=0
        marginal_likelihood_s[idx] = likelihood(data_E, 0, 0)
        marginal_error_s[idx] = 0  # No integration, so error is zero

        # Nonstationary: theta=0, marginalize over delta
        def integrand(delta, lambda2):
            return likelihood(data_E, 0, delta) * prior_1d(delta, lambda2)
        
        integral, error = marginalize_likelihood_1d_trapezoidal(data_E, delta_range, lambda_=lambda2, integrand=integrand)

        """
        integral, error = quad(
            integrand,
            delta_range[0], delta_range[1],
            epsrel=INTEGRATION_TOL, epsabs=INTEGRATION_TOL
        )
        """

        marginal_likelihood_ns[idx] = integral
        marginal_error_ns[idx] = error

    marginal_likelihood_s = np.dot(marginal_likelihood_s, prior_E(embedding_dimensions, p))
    marginal_error_s = np.sum(marginal_error_s)
    marginal_likelihood_ns = np.dot(marginal_likelihood_ns, prior_E(embedding_dimensions, p))
    marginal_error_ns = np.sum(marginal_error_ns)

    # Compute the Bayes Factor
    bayes_factor = marginal_likelihood_ns / marginal_likelihood_s
    error_bf = marginal_error_s / marginal_likelihood_s + marginal_error_ns / marginal_likelihood_ns
    if debug:
        return bayes_factor, error_bf, marginal_likelihood_s, marginal_error_s, marginal_likelihood_ns, marginal_error_ns
    else:
        # Return the Bayes Factor and error estimate
        return bayes_factor, error_bf

# Function to perform the nonstationarity test with autoregressive model structure
# Meant to emulate classical nonstationarity tests like Dickey-Fuller
# Inputs:
def nonstationarity_test_linear(data, delta_range=(0.0, 4.0), E_range=(0, 8), lambda1=1.0, p=0.5):
    # Compute the Bayes Factor for the linear case
    bayes_factor, error_bf = compute_bayes_factor_linear(data, delta_range, E_range, lambda1, p, False)

    # Compute the log Bayes Factor
    evidence = 10 * np.log10(bayes_factor)

    # Compute the significance level
    significance_level = 1 - (10 ** (-evidence/10))

    return evidence, significance_level, error_bf
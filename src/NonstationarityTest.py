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

    # Support scalar or array input for theta and delta
    if np.isscalar(theta) and np.isscalar(delta):
        log_likelihood = ns.logLikelihood(Xemb, Y, tx, theta, delta)
        return np.exp(log_likelihood)
    else:
        # Broadcast theta and delta to same shape
        theta = np.atleast_1d(theta)
        delta = np.atleast_1d(delta)
        # If both are 1D and same length, treat as pairs
        if theta.shape == delta.shape:
            log_likelihoods = np.array([ns.logLikelihood(Xemb, Y, tx, th, dl) for th, dl in zip(theta, delta)])
        else:
            # If one is scalar, broadcast
            if theta.size == 1:
                log_likelihoods = np.array([ns.logLikelihood(Xemb, Y, tx, theta.item(), dl) for dl in delta])
            elif delta.size == 1:
                log_likelihoods = np.array([ns.logLikelihood(Xemb, Y, tx, th, delta.item()) for th in theta])
            else:
                # If both are arrays but different shapes, use meshgrid
                theta_grid, delta_grid = np.meshgrid(theta, delta, indexing='ij')
                log_likelihoods = np.empty(theta_grid.shape)
                for i in range(theta_grid.shape[0]):
                    for j in range(theta_grid.shape[1]):
                        log_likelihoods[i, j] = ns.logLikelihood(Xemb, Y, tx, theta_grid[i, j], delta_grid[i, j])
        # Numerical stability: shift by max
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        return np.exp(log_likelihoods)

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

def posterior_1d_linear(param1, data, lambda1=1.0):
    return likelihood(data, 0, 0) * prior_1d(param1, lambda1)
"""

def posterior_1d_linear(param1, data, lambda1=1.0):
    # Vectorize the computation
    vectorized_likelihood = np.vectorize(lambda p: likelihood(data, 0, p))
    vectorized_prior = np.vectorize(lambda p: prior_1d(p, lambda1))
    return vectorized_likelihood(param1) * vectorized_prior(param1)

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
    likelihood_vals = likelihood(data, param1_flat, param2_flat)
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

def marginalize_likelihood_1d_trapezoidal(data, param_range, lambda_=1.0, integrand=posterior_1d, resolution=10):
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
    posterior_vals = np.nan_to_num(integrand(param_vals, data, lambda_), nan=0.0)
    integral = np.trapz(posterior_vals, x=param_vals)

    return integral, None

# Shared helper for grid-based marginalization
def _marginalize_grid(log_likelihoods, prior_grid, x_axes, sum_axis):
    log_likelihoods = log_likelihoods - np.max(log_likelihoods)
    likelihoods = np.exp(log_likelihoods)
    posterior_grid = likelihoods * prior_grid
    for axis, x in x_axes:
        posterior_grid = np.trapz(posterior_grid, x=x, axis=axis)
    marginal = np.sum(posterior_grid, axis=sum_axis)
    return marginal

# Build log likelihood grid for E, theta, delta (or delta=0 for S-map)
def _build_log_likelihood_grid(Xr, tx, E_range, theta_range, delta_range, tau, delta_is_zero=False):
    E_vals = np.arange(E_range[0], E_range[1]+1)
    theta_vals = np.array(theta_range)
    delta_vals = np.array(delta_range) if not delta_is_zero else np.array([0])
    grid_shape = (len(E_vals), len(theta_vals), len(delta_vals))
    log_likelihoods = np.empty(grid_shape)
    for i, E in enumerate(E_vals):
        Xemb, Y, tx_ = ns.delayEmbed(Xr, E, tau, t=tx)
        for j, th in enumerate(theta_vals):
            for k, dl in enumerate(delta_vals):
                # average log likelihood per data point
                log_likelihoods[i, j, k] = ns.logLikelihood(Xemb, Y, tx_, th, dl)
    # Normalize by shortest length to account for varying data sizes
    Xemb, Y, tx_ = ns.delayEmbed(Xr, E_range[1], tau, t=tx)
    shortest_len = len(Y)
    log_likelihoods *= shortest_len
    return log_likelihoods, E_vals, theta_vals, delta_vals

# Marginalize over E, theta, delta (NSMap)
def marginalize_likelihood_nsmap_trapezoidal(data, E_range, theta_range, delta_range, lambda1=1.0, lambda2=1.0, p_E=0.5):
    Xr, tx, tau = data
    log_likelihoods, E_vals, theta_vals, delta_vals = _build_log_likelihood_grid(Xr, tx, E_range, theta_range, delta_range, tau)
    prior_E_vals = np.array([prior_E(E, p_E) for E in E_vals])
    prior_theta_vals = np.array([prior_1d(th, lambda1) for th in theta_vals])
    prior_delta_vals = np.array([prior_1d(dl, lambda2) for dl in delta_vals])
    prior_grid = prior_E_vals[:, None, None] * prior_theta_vals[None, :, None] * prior_delta_vals[None, None, :]
    trapz_axes = [(1, theta_vals), (2, delta_vals)]
    marginal = _marginalize_grid(log_likelihoods, prior_grid, trapz_axes, sum_axis=0)
    return marginal, None

# Marginalize over E and theta (SMap)
def marginalize_likelihood_smap_trapezoidal(data, E_range, theta_range, lambda1=1.0, p_E=0.5):
    Xr, tx, tau = data
    log_likelihoods, E_vals, theta_vals, _ = _build_log_likelihood_grid(Xr, tx, E_range, theta_range, [0], tau, delta_is_zero=True)
    prior_E_vals = np.array([prior_E(E, p_E) for E in E_vals])
    prior_theta_vals = np.array([prior_1d(th, lambda1) for th in theta_vals])
    prior_grid = prior_E_vals[:, None] * prior_theta_vals[None, :]
    trapz_axes = [(1, theta_vals)]
    marginal = _marginalize_grid(log_likelihoods.squeeze(), prior_grid, trapz_axes, sum_axis=0)
    return marginal, None

def marginalize_likelihood_E(marginal_likelihood_func, data, param_range, lambda_=1.0, p=0.5):
    marginal_likelihood = 0
    marginal_error = 0
    for E in range(param_range[0], param_range[1]):
        integral, error = marginal_likelihood_func(data, E, lambda_, p) * prior_E(E)
        marginal_likelihood += integral
        marginal_error += error
    return marginal_likelihood, marginal_error

def compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5, debug=False, resolution=40, return_best_params=False):

    """
    Compute the Bayes Factor between stationary and nonstationary models.
    """

    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    delta_vals = np.linspace(delta_range[0], delta_range[1], resolution)
    E_vals = np.arange(E_range[0], E_range[1]+1)

    # Build log likelihood grids for both NSMap and SMap
    Xr, tx, tau = data
    log_lik_nsmap, E_vals, theta_vals, delta_vals = _build_log_likelihood_grid(Xr, tx, E_range, theta_vals, delta_vals, tau)
    log_lik_smap, _, _, _ = _build_log_likelihood_grid(Xr, tx, E_range, theta_vals, [0], tau, delta_is_zero=True)

    # Compute the maximum log likelihood over both grids
    max_log_likelihood = np.maximum(log_lik_nsmap, log_lik_smap).max()

    # Priors
    prior_E_vals = np.array([prior_E(E, p) for E in E_vals])
    prior_theta_vals = np.array([prior_1d(th, lambda1) for th in theta_vals])
    prior_delta_vals = np.array([prior_1d(dl, lambda2) for dl in delta_vals])
    prior_grid_nsmap = prior_E_vals[:, None, None] * prior_theta_vals[None, :, None] * prior_delta_vals[None, None, :]
    prior_grid_smap = prior_E_vals[:, None] * prior_theta_vals[None, :]

    # Posterior grids (normalized for numerical stability)
    post_grid_nsmap = np.exp(log_lik_nsmap - max_log_likelihood) * prior_grid_nsmap
    post_grid_smap = np.exp(log_lik_smap.squeeze() - max_log_likelihood) * prior_grid_smap

    # Marginalize over all parameters
    marginal_nsmap = np.trapz(np.trapz(post_grid_nsmap, x=theta_vals, axis=1), x=delta_vals, axis=1)
    marginal_nsmap = np.sum(marginal_nsmap)
    marginal_smap = np.trapz(post_grid_smap, x=theta_vals, axis=1)
    marginal_smap = np.sum(marginal_smap)

    # Normalize
    normalization = marginal_nsmap + marginal_smap
    if normalization == 0:
        normalization = 1e-12
    marginal_nsmap /= normalization
    marginal_smap /= normalization

    bayes_factor = marginal_nsmap / marginal_smap

    if debug:
        return bayes_factor, marginal_smap, marginal_nsmap
    elif return_best_params:
        best_idx_ns = np.unravel_index(np.argmax(post_grid_nsmap), post_grid_nsmap.shape)
        best_E = E_vals[best_idx_ns[0]]
        best_theta = theta_vals[best_idx_ns[1]]
        best_delta = delta_vals[best_idx_ns[2]]
        return bayes_factor, (best_theta, best_delta, best_E)
    else:
        return bayes_factor

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
def nonstationarity_test(data, theta_range=(0.0, 4.0), delta_range=(0.0, 4.0), E_range=(0, 8), lambda1=1.0, lambda2=1.0, p=0.5, resolution = 40, return_best_params=False):
    # Compute the Bayes Factor
    if return_best_params:
        bayes_factor, best_params = compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1=lambda1, lambda2=lambda2, p=p, resolution=resolution, debug=False, return_best_params=return_best_params)
    else:
        bayes_factor = compute_bayes_factor(data, theta_range, delta_range, E_range, lambda1=lambda1, lambda2=lambda2, p=p, resolution=resolution, debug=False, return_best_params=return_best_params)

    # Compute the log Bayes Factor
    evidence = 10 * np.log10(bayes_factor)

    # Compute the significance level
    significance_level = bayes_factor / (bayes_factor + 1)
    # significance_level = 1 - (10 ** (-evidence/10))

    if return_best_params:
        return evidence, significance_level, best_params
    
    return evidence, significance_level

def compute_bayes_factor_linear(data, delta_range=(0.0, 4.0), E_range=(0, 8), lambda2=1.0, p=0.5, debug=False, resolution=20, return_best_params=False):
    # Use _build_log_likelihood_grid to build posterior grids for linear case
    Xr, tx, tau = data
    E_vals = np.arange(E_range[0], E_range[1])
    delta_vals = np.linspace(delta_range[0], delta_range[1], resolution)
    theta_vals = np.array([0])  # Linear case: theta=0

    # Nonstationary: theta=0, delta varies
    log_lik_ns, E_grid, _, delta_grid = _build_log_likelihood_grid(Xr, tx, (E_range[0], E_range[1]-1), theta_vals, delta_vals, tau)
    # Stationary: theta=0, delta=0
    log_lik_s, E_grid_s, _, _ = _build_log_likelihood_grid(Xr, tx, (E_range[0], E_range[1]-1), theta_vals, [0], tau, delta_is_zero=True)

    # Priors
    prior_E_vals = np.array([prior_E(E, p) for E in E_vals])
    prior_delta_vals = np.array([prior_1d(dl, lambda2) for dl in delta_vals])
    prior_grid_ns = prior_E_vals[:, None] * prior_delta_vals[None, :]
    prior_grid_s = prior_E_vals

    # Numerical stability
    max_log_likelihood = max(np.max(log_lik_ns), np.max(log_lik_s))

    # Posterior grids
    post_grid_ns = np.exp(log_lik_ns.squeeze() - max_log_likelihood) * prior_grid_ns
    post_grid_s = np.exp(log_lik_s.squeeze() - max_log_likelihood) * prior_grid_s

    # Marginalize
    marginal_ns = np.trapz(post_grid_ns, x=delta_vals, axis=1)
    marginal_ns = np.sum(marginal_ns)
    marginal_s = np.sum(post_grid_s)

    # Normalize
    normalization = marginal_ns + marginal_s
    if normalization == 0:
        normalization = 1e-12
    marginal_ns /= normalization
    marginal_s /= normalization

    bayes_factor = marginal_ns / marginal_s

    if debug:
        return bayes_factor, marginal_s, marginal_ns
    elif return_best_params:
        # Find the index of the maximum posterior in the NS grid
        best_idx = np.unravel_index(np.argmax(post_grid_ns), post_grid_ns.shape)
        best_E = E_vals[best_idx[0]]
        best_delta = delta_vals[best_idx[1]]
        return bayes_factor, (best_delta, best_E)
    else:
        return bayes_factor

# Function to perform the nonstationarity test with autoregressive model structure
# Meant to emulate classical nonstationarity tests like Dickey-Fuller
# Inputs:
def nonstationarity_test_linear(data, delta_range=(0.0, 4.0), E_range=(0, 8), lambda2=1.0, p=0.5, resolution=20, return_best_params=False):
    # Compute the Bayes Factor for the linear case
    if return_best_params:
        bayes_factor, best_params = compute_bayes_factor_linear(data, delta_range=delta_range, E_range=E_range, lambda2=lambda2, p=p, debug=False, resolution=resolution, return_best_params=return_best_params)
    else:
        bayes_factor = compute_bayes_factor_linear(data, delta_range=delta_range, E_range=E_range, lambda2=lambda2, p=p, debug=False, resolution=resolution)

    # Compute the log Bayes Factor
    evidence = 10 * np.log10(bayes_factor)

    # Compute the significance level
    significance_level = bayes_factor / (bayes_factor + 1)

    if return_best_params:
        return evidence, significance_level, best_params
    
    return evidence, significance_level
import src.NSMap as ns
import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad
from src.NonstationarityTest import prior_E, posterior_2d
import src.NonstationarityTest as nt

def compute_posterior_weighted_parameters(data, theta_range, delta_range, E_range, lambda1=1.0, lambda2=1.0, p=0.5, resolution=20):
    """
    Compute the posterior-weighted mean of theta and delta, marginalizing over E, theta, delta with numerically stable grid integration.
    Args:
        data: tuple (Xr, tx, tau)
        theta_range: (min, max)
        delta_range: (min, max)
        E_range: (min, max)
        lambda1, lambda2: prior parameters
        p: prior parameter for E
        resolution: grid resolution
    Returns:
        (mean_theta, mean_delta)
    """
    Xr, tx, tau = data
    # Compute the marginalized posterior grid over theta, delta (marginalizing E)
    marginalized_posterior, theta_values, delta_values = compute_posterior_on_grid(
        (Xr, tx, tau), theta_range, delta_range, lambda1=lambda1, lambda2=lambda2, resolution=resolution, E_range=E_range, p=p
    )
    # Normalize posterior
    Z = np.sum(marginalized_posterior)
    if Z == 0:
        raise ValueError("Posterior normalization constant is zero.")
    posterior_norm = marginalized_posterior / Z

    # Compute grid for theta and delta
    theta_grid, delta_grid = np.meshgrid(theta_values, delta_values, indexing='ij')

    # Compute posterior-weighted means
    mean_theta = np.sum(posterior_norm * theta_grid)
    mean_delta = np.sum(posterior_norm * delta_grid)
    return mean_theta, mean_delta

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

def compute_posterior_on_grid(data, param1_range, param2_range, E_range=[0, 3], lambda1=1.0, lambda2=1.0, resolution=10, p=0.5):
    # Use _build_log_likelihood_grid for numerically stable log likelihood computation over all E
    # data: (Xr, tx, tau)
    if len(data) == 4:
        raise ValueError("E_range must be provided for marginalization over E.")
    elif len(data) == 3:
        Xr, tx, tau = data
    
    theta_values = np.linspace(param1_range[0], param1_range[1], resolution)
    delta_values = np.linspace(param2_range[0], param2_range[1], resolution)

    # Build log-likelihood grid over all E, theta, delta
    log_likelihoods, E_vals, theta_vals, delta_vals = nt._build_log_likelihood_grid(
        Xr, tx, E_range, theta_values, delta_values, tau)
    
    # Build prior grid: shape (len(E_vals), len(theta_vals), len(delta_vals))
    prior_E_vals = np.array([nt.prior_E(E, p) for E in E_vals])
    prior_theta_vals = np.array([nt.prior_1d(th, lambda1) for th in theta_vals])
    prior_delta_vals = np.array([nt.prior_1d(dl, lambda2) for dl in delta_vals])
    prior_grid = prior_E_vals[:, None, None] * prior_theta_vals[None, :, None] * prior_delta_vals[None, None, :]
    
    # Normalize log-likelihoods for stability
    log_likelihoods = log_likelihoods - np.max(log_likelihoods)
    
    # Compute unnormalized posterior grid
    posterior_grid = np.exp(log_likelihoods) * prior_grid
    
    # Marginalize (sum/integrate) over E axis (axis=0)
    marginalized_posterior = np.sum(posterior_grid, axis=0)
    dtheta = (param1_range[1] - param1_range[0]) / (resolution - 1)
    ddelta = (param2_range[1] - param2_range[0]) / (resolution - 1)
    marginalized_posterior *= dtheta * ddelta

    return marginalized_posterior, theta_values, delta_values

def posterior_weighted_function_grid(func, data, param1_range, param2_range, lambda1=1.0, lambda2=1.0, resolution=10):
    theta_values = np.linspace(param1_range[0], param1_range[1], resolution)
    delta_values = np.linspace(param2_range[0], param2_range[1], resolution)

    theta, delta = np.meshgrid(theta_values, delta_values, indexing='ij')

    post_weighted_func = np.nan_to_num(posterior_2d(theta_values, delta_values, data, lambda1, lambda2) * func(theta, delta), 0)

    dtheta = (param1_range[1] - param1_range[0]) / (resolution - 1)
    ddelta = (param2_range[1] - param2_range[0]) / (resolution - 1)

    integral = np.sum(post_weighted_func) * dtheta * ddelta

    return integral
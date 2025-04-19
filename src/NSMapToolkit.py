import numpy as np
import numpy.linalg as la
from scipy.integrate import dblquad
from scipy.integrate import quad
from tqdm.notebook import tqdm

# Simple function which takes a time series with shape (n,) and return it with
# mean 0 and standard deviation 1
#   Parameters
#       x - vector time series input
#   Results
#       x with mean subtracted and standard deviation of 1
def standardize(x):
    return (x - np.mean(x, axis=0, where=np.isfinite(x))) / np.std(x, axis=0, where=np.isfinite(x))

# Create a delay embeddding vector from a given UNIVARIATE time series.
#   Parameters
#       D - a univariate time series with vector shape (n,)
#       E - number of columns in training data matrix X
#       tau - number of steps between each column
#       t (optional) - if sampling is non-uniform, then specify the time
#                      of each training data point, standardized between 0 and 1,
#                      a correctly embedded version of t will also be returned
#       removeNAs (optional) - if true, then remove all rows in full embedding 
#                              matrix with null value
#   Returns
#       (X, Y) - (training data matrix of shape (n-tau*E, E), target values of shape (n, 1))
#           OR
#       (X, Y, t) - (training data matrix of shape (n-tau*E, E),
#                    target values of shape (n, 1),
#                    time vector updated correctly)
def delayEmbed(D, E, tau, t = None, removeNAs=True):
    
    n = D.shape[0]
    # the time series and time index t must have the same length!
    if t is not None:
        assert n == len(t)

    # A is the delay matrix with padded 0s at the top and bottom
    
    totalRows = n + tau * E
    A = np.zeros((totalRows, E + 1))

    for i in range(E + 1):
        lower = i * tau
        upper = lower + n
        A[lower:upper, i] = D.flatten()
    
    # B is A with all padded rows removed
    rowsLost = E * tau
    if rowsLost != 0:
        B = A[rowsLost : -rowsLost]
        # if t exists, make it line up with the chronologically latest column of X
        if t is not None:
            tTemp = np.zeros(totalRows)
            tTemp[tau: tau + n] = t
            t = tTemp[rowsLost : -rowsLost]
    else:
        B = A
    
    # remove all rows containing any null values, if requested
    if removeNAs:
        notNA = np.all(~np.isnan(B),axis=1)

        B = B[notNA]
        if t is not None:
            t = t[notNA]
    
    # if a custom t is specified, then return the correct version alongside it
    if t is None:
        return (B[:,1:], B[:,0, None])
    else:
        return (B[:,1:], B[:,0, None], t)

# Simple function which computes the hat matrix.
#   Parameters
#       M - time delay embedding matrix augmented with a column of 1s
#       W - diagonal (n,n) matrix of weights
#       x - E+1 dimensional vector with 1 in final entry, represents current
#           state from which to make a prediction
def getHat(M, W, x):
    hat = x @ la.pinv(W@M) @ W
    return hat

# Compute the trace of the hat matrix where all training data are included (this
# is not cross validation). This is a canonical estimation of model degrees of
# freedom for local linear models.
#   Parameters
#       X - embedded training data, output of delayEmbed
#       Y - vector of targets, output of delayEmbed
#       tx - the times for each data point, normalized between 0 and 1
#       theta - value for hyperparameter theta
#       delta - value for hyperparameter delta
#   Returns
#       dofest - estimation of the degrees of freedom of a model for this dataset
def dofestimation(X, Y, tx, theta, delta):
    dofest = 0
    # for each entry in the training dataset, compute the hat vector, then add
    # the ith entry to our running total to compute the trace of the whole matrix
    for i in range(X.shape[0]):
        pred, hatvector = NSMap(X, Y, tx, X[i], tx[i], theta, delta, return_hat=True)
        dofest += hatvector[i]
    return dofest

# Leaves one input and output pair out, and use rest as training data
# returns predictions which are the length of the whole time series
#   Parameters
#       X - embedded training data, output of delayEmbed
#       Y - vector of targets, output of delayEmbed
#       tx - the times for each data point, normalized between 0 and 1
#       theta - value for hyperparameter theta
#       delta - value for hyperparameter delta
#       get_hat (optional) - if True, then hat matrix
#   Returns
#       timestepPredictions - vector of predictions for each training input
#           OR
#       (timestepPredictions, hat) - (vector of predictions for each training input,
#                                     hat matrix)
def leaveOneOut(X, Y, tx, theta, delta, get_hat=False):
    
    if get_hat:
        hat = np.zeros((X.shape[0], X.shape[0]-1))
    timestepPredictions = np.zeros((X.shape[0], 1))
    
    for i in range(0, X.shape[0]):
        # create the train and test stuff
        
        Xjts = X[i].copy()
        Yjts = Y[i].copy()
        tXjts = tx[i].copy()
        
        Xjtr = np.delete(X, i, axis=0)
        Yjtr = np.delete(Y, i, axis=0)
        tXjtr = np.delete(tx, i, axis=0)
        
        if get_hat:
            prediction, hat_vector = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=True)
            hat[i,:] = hat_vector
        else:
            prediction = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=False)
        
        timestepPredictions[i] = prediction
            
    if get_hat:
        return (timestepPredictions, hat)
    else:
        return timestepPredictions

# Computes the log likelihood for given hyperparameters, training and target
# data.
#   Parameters
#       X - embedded training data, output of delayEmbed
#       Y - vector of targets, output of delayEmbed
#       tx - the times for each data point, normalized between 0 and 1
#       theta - value for hyperparameter theta
#       delta - value for hyperparameter delta
#       returnSeries (optional) - if True, then return predictions for each element 
#                               of training set
#   Returns
#       lnL - scalar log likelihood
#           OR
#       (lnL, Yhat) - (scalar log likelihood, predictions)
def logLikelihood(X, Y, tx, theta, delta, returnSeries=False):
    
    n = Y.shape[0]

    Yhat = leaveOneOut(X, Y, tx, theta, delta)
    
    k = dofestimation(X, Y, tx, theta, delta)
    mean_squared_residuals = np.sum((Y-Yhat)**2) / (n-k)

    lnL = (-n/2)*(np.log(mean_squared_residuals) + np.log(2*np.pi) + 1 )

    # bic = BIC(lnL, k, n)
 
    if returnSeries:
        return (lnL, Yhat)
    else:
        return lnL

    # temporary substitution for -BIC instead of unbiased lnL
    # if returnSeries:
    #     return (-bic, Yhat)
    # else:
    #     return -bic

# Alternate to log likelihood, compute the BIC instead
def BIC(logLikelihood, num_parameters, sample_size):
    bic = -2 * logLikelihood + num_parameters * np.sqrt(sample_size)
    return bic

# Simple implementation of the classic SMap. Does the same thing as NSMap
# with delta=0.
#   Parameters
#       X - (ndarray) training data, (n,p) array of state space variables
#       Y - (ndarray) labels
#       x - (ndarray) current state to predict from
#       theta - (scalar) hyperparameter
#   Returns
#       scalar prediction
def SMap(X, Y, x, theta):
    norms = la.norm(X-x,axis=1)
    d = np.mean(norms) # d = np.mean(norms) # 
    
    W = np.diag(np.exp(-1 * theta * norms / d))

    H = getHat(X, W, x)
    return H @ Y

# Implementation of NSMap! Note that T and t(where) must be standardized 
# to be between 0 and 1.
#   Parameters
#       X - (ndarray) training data, (n,p) array of state space variables
#       Y - (ndarray) labels
#       T - (ndarray) time for each row in X
#       x - (ndarray) current state to predict from
#       t - (scalar) current time to predict from
#       theta - (scalar) hyperparameter
#       delta - (scalar) hyperparameter
#   Returns
#       scalar prediction
#           OR
#       (scalar prediction, hat matrix row) if return_hat is true
#           OR
#       (scalar prediction, hat matrix row, derivative of h wrt theta, derivative of h wrt delta)
#               if return_hat_derivatives is True
def NSMap(X, Y, T, x, t, theta, delta, return_hat=False, return_hat_derivatives=False):
    
    n = X.shape[0]
    norms = la.norm(X - x,axis=1)
    d = np.mean(norms)

    W = np.exp(-1*(theta*norms)/d - delta*(T-t)**2)[:,None] if d != 0 else np.exp(- delta*(T-t)**2)[:,None]

    # augment the training data with a column of 1s to allow for intercepts
    M = np.hstack([X, np.ones((n,1))])
    xaug = np.hstack([x, 1]).T

    if return_hat or return_hat_derivatives:
        # weighted penrose inverse
        pinv = la.pinv(W*M)

        H = xaug @ (pinv.T * W).T
        prediction = (H @ Y)[0]

        # compute the derivatives of the hat matrix wrt theta and delta
        if return_hat_derivatives:
            dWdtheta = -1 * W.flatten() * norms / d if d != 0 else 0 * W.flatten()
            dWddelta = -1 * W.flatten() * ((T-t)**2)

            dthetapinv = (dWdtheta[:,None].T * pinv)
            ddeltapinv = (dWddelta[:,None].T * pinv)

            dhdtheta = 2 * xaug @ (dthetapinv - dthetapinv @ M @ (pinv * W.T))
            dhddelta = 2 * xaug @ (ddeltapinv - ddeltapinv @ M @ (pinv * W.T))

            return (prediction, H, dhdtheta, dhddelta)
    
        return (prediction, H)
    else:
        # use least squares to solve if hat matrix derivates aren't needed,
        # as this is much faster than computing the penrose inverse
        # and gives the same output
        prediction = xaug @ la.lstsq( W * M, W * Y, rcond=None)[0]
        return prediction

# The test of nonstationarity for a given univariate series. Calculates aggregated 
# delta and theta, and r_sqrd optionally.
#   Parameters
#       Xr (ndarray)- univariate input time series in vector shape (n,)
#       E - the number of columns in the delay matrix, same definition as in the paper
#       tau - number of steps between lags, default value is 1
#       t - vector representing the time of each observation, must begin at 0 and end
#           at 1. A default t will be create which assumes equal spacing between inputs
#           if one isn't specified here.
#       trainingSteps - maximum number of gradient descent steps before optimization terminates
#       return_forecast_skill - if True, then then return maximum forecast r^2 attained
#                               across all embedding dimensions
#       theta_fixed - if True, then hold theta at 0, forcing linear model structure.
#                     This is used in round 1 of experiments to determine if nonlinear
#                     model structure is needed to detect nonstationarity.
def get_delta_agg(Xr, E, tau=1, t=None, trainingSteps=100, return_forecast_skill=False, theta_fixed=False):

    if t is None:
        t = np.linspace(0,1, num=len(Xr))
    else:
        # Remember to standardize t to be between 0 and 1!
        assert t[0] == 0 and t[-1] == 1

    # produce delay embedding vector first so the set of targets is fixed across all E
    Xemb, Y, tx = delayEmbed(Xr, E, tau, t=t)

    # compute optimal hyperparameters and likelihood for NS-Map and S-Map 
    # for each number of lags from 1 to E
    table = np.zeros((E, 5))
    hp = np.zeros(2)
    for l in range(1, E + 1):
        X = Xemb[:, :l]

        # find optimal theta and delta using rprop gradient descent and their log likelihoods
        thetaNS, deltaNS, lnLNS = optimizeG(X, Y, tx, fixed=np.array([theta_fixed, False]), trainingSteps=trainingSteps, hp=hp.copy())
        # find optimal theta for NSMap using the same algorithm, holding theta fixed
        thetaS, _, lnLS = optimizeG(X, Y, tx, fixed=np.array([theta_fixed, True]),trainingSteps=trainingSteps, hp=hp.copy())

        table[l - 1] = np.array([deltaNS, lnLNS, lnLS, thetaNS, thetaS])

    # Aggregate delta using the different in likelihood of S-Map and NS-Map
    lnLdifference = table[:,1] - table[:,2]
    delta_agg_weights = np.exp(lnLdifference - np.max(lnLdifference))
    delta_agg = np.average(table[:,0], weights=delta_agg_weights)

    # return the theta which maximizes log likelihood of NSMap
    theta = table[np.argsort(table[:,1])[-1], 3]

    if return_forecast_skill:
        return (delta_agg, theta, get_r_sqrd(table, Xemb, Y, tau, tx))
    else: 
        return delta_agg

# Compute the r squared coefficient based on the other data from get_delta_agg
def get_r_sqrd(table, Xemb, Y, tau, tx):
    ibestNS = np.argmax(table[:,1])
    ibestS = np.argmax(table[:,2])

    # if NSMap has a higher log likelihood than SMap then use NSMap's hyperparameters
    if table[ibestNS,1] > table[ibestS,2]:
        delta = table[ibestNS, 0]
        theta = table[ibestNS, 3]
        i = ibestNS
    # else use SMap's
    else:
        delta = 0
        theta = table[ibestS, 4]
        i = ibestS

    # produce forecasts based on the optimal hyperparameters
    X = Xemb[:,:(i+1)*tau:tau]
    Y_hat = leaveOneOut(X, Y, tx, theta, delta)

    rsqr = np.corrcoef(Y.flatten(), Y_hat.flatten())[0,1] ** 2

    return rsqr

"""
# finds the gradient of the likelihood function with respect to our hyperparameters theta and delta
def gradient(X, Y, tx, theta, delta):
    # we should be able to pull this off with two passes, once for leave one out and again leave all in.

    n = X.shape[0]
    
    dSSE_dtheta = 0
    dDOF_dtheta = 0
    dSSE_ddelta = 0
    dDOF_ddelta = 0

    SSE = 0
    dof = 0

    for i in range(0, X.shape[0]):
        Xjts = X[i].copy()
        Yjts = Y[i].copy()
        tXjts = tx[i].copy()
        
        Xjtr = np.delete(X, i, axis=0)
        Yjtr = np.delete(Y, i, axis=0)
        tXjtr = np.delete(tx, i, axis=0)
    
        prediction, _, hat_vec_dtheta_L, hat_vec_ddelta_L = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat_derivatives=True)
        _, hat_vec, hat_vec_dtheta, hat_vec_ddelta = NSMap(X, Y, tx, Xjts, tXjts, theta, delta, return_hat_derivatives=True)

        residual = Yjts[0] - prediction

        SSE += (residual) ** 2
        dof += hat_vec[i]

        dSSE_dtheta += -2 * residual * (hat_vec_dtheta_L @ Yjtr)
        dSSE_ddelta += -2 * residual * (hat_vec_ddelta_L @ Yjtr)
        dDOF_dtheta += hat_vec_dtheta[i]
        dDOF_ddelta += hat_vec_ddelta[i]

    assert type(SSE) == np.float64

    # this is ugly, but we have to include the max stuff to prevent divide by 0 errors
    dl_dtheta = (-n/2) * ( dSSE_dtheta / max(SSE, 10e-6) + dDOF_dtheta / max(n-dof, 10e-6))
    dl_ddelta = (-n/2) * ( dSSE_ddelta / max(SSE, 10e-6) + dDOF_ddelta / max(n-dof, 10e-6))

    E = ((-n/2) * ( np.log(max(SSE, 10e-6) / max(n-dof, 10e-6)) + np.log(2*np.pi) + 1))

    return (np.hstack([dl_dtheta, dl_ddelta]), E)

"""
# Simple implementation using the triangle method
def gradient(X, Y, tx, theta, delta):
    # we should be able to pull this off with two passes, once for leave one out and again leave all in.

    n = X.shape[0]
    dp = 10e-2 # differential for parameters

    lnL = logLikelihood(X, Y, tx, theta, delta)
    dtheta = logLikelihood(X, Y, tx, theta + dp, delta)
    ddelta = logLikelihood(X, Y, tx, theta, delta + dp)

    dlnL_dtheta = (dtheta - lnL) / dp
    dlnL_ddelta = (ddelta - lnL) / dp

    return (np.hstack([dlnL_dtheta, dlnL_ddelta]), lnL)

"""
# Find the optimal hyperparameters of NS-Map or S-Map using gradient descent
def optimizeG(X, Y, t, trainingSteps=40, hp=np.array([0.0,0.0]), fixed=np.array([False, False])):    
    err = 0
    
    gradPrev = np.ones(hp.shape, dtype=float)
    deltaPrev = np.ones(hp.shape, dtype=float)
    
    for count in range(trainingSteps):
        errPrev = err
        grad, err = gradient(X, Y, t, hp[0], hp[1])

        print(grad)

        if abs(err-errPrev) < 0.01 or count == trainingSteps-1:
            break

        dweights, deltaPrev, gradPrev = calculateHPChange(grad, gradPrev, deltaPrev)

        # floor and ceiling the hyperparameters
        for i in range(2):
            if not fixed[i]:
                hp[i] = max(0, hp[i] + dweights[i])

    return (hp[0], hp[1], err)
"""

def optimizeG(X, Y, t, trainingSteps=40, hp=np.array([0.0,0.0]), fixed=np.array([False, False])):
    theta_max = 0
    delta_max = 0
    lnL_max = float('-inf')

    for theta in range(10):
        for delta in range(10):
            current_lnL = logLikelihood(X, Y, t, theta, delta)
            if current_lnL > lnL_max:
                theta_max = theta
                delta_max = delta
                lnL_max = current_lnL 

    return (theta_max, delta_max, lnL_max)

def calculateHPChange(grad, gradPrev, deltaPrev):
    rhoplus = 1.2 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1
    
    grad = grad / la.norm(grad) # NORMALIZE, because rprop ignores magnitude

    s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
    spos = np.ceil(s) # 0 for - vals, 1 for + vals
    sneg = -1 * (spos - 1)

    delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
    dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

    return (dweights, delta, grad)


## Bayesian Nonstationarity Detection

# Define the likelihood function
def likelihood(data, theta, delta):
    (Xr, tx, E, tau) = data

    Xemb, Y, tx = delayEmbed(Xr, E, tau, t=tx)

    # Replace this with the actual likelihood function
    # This is just a placeholder example
    return np.exp(logLikelihood(Xemb, Y, tx, theta, delta))

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
        epsrel=1e-3,
        epsabs=1e-3
    )
    return integral, error

# Function to marginalize the posterior over the parameter space
def marginalize_likelihood_2d(data, param1_range, param2_range, lambda1=1.0, lambda2=1.0):
    integral, error = dblquad(
        posterior_2d,
        param1_range[0], param1_range[1],  # Integration limits for param1
        lambda param1: param2_range[0], lambda param1: param2_range[1],  # Integration limits for param2
        args=(data, lambda1, lambda2),
        epsrel=1e-3,
        epsabs=1e-3
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

    for E in tqdm(range(E_range[0], E_range[1])):
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
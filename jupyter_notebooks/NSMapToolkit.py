import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modelSystems import *

def standardize(x):
    return (x - np.mean(x, axis=0, where=np.isfinite(x))) / np.std(x, axis=0, where=np.isfinite(x))

# create a delay embeddding vector from a given UNIVARIATE time series.
def delayEmbed(D, predHorizon, nLags, embInterval, t = None, removeNAs=True):
    
    totalRows = D.shape[0] + predHorizon + embInterval * nLags
    A = np.zeros((totalRows, 2 + nLags))
    
    A[:D.shape[0],0] = D.flatten()
    
    for i in range(1, 2 + nLags):
        lower = predHorizon + (i - 1) * embInterval
        upper = lower + D.shape[0]
        A[lower:upper, i] = D.flatten()
    
    rowsLost = predHorizon + nLags * embInterval
    if rowsLost != 0:
        B = A[rowsLost : -rowsLost]
        if t is not None:
            t = t[ : -rowsLost]
    else: 
        B = A
    
    if removeNAs:
        notNA = np.all(~np.isnan(B),axis=1)

        B = B[notNA]
        if t is not None:
            # print(t.shape, notNA.shape)
            t = t[notNA]
    
    if t is None:
        return (B[:,1:], B[:,0, None])
    else:
        return (B[:,1:], B[:,0, None], t)

def getHat(M, W, x):
    hat = x @ la.pinv(W@M) @ W
    return hat

# WRONG, NEED TO USE APPROPRIATE HAT MATRIX, WHICH IS MADE OF 
def dofestimation(X, Y, tx, theta, delta):

    #_, hat = leaveOneOut(X, Y, tx, theta, delta,get_hat=True)
    # print(hat.shape)
    #dofest = np.trace(hat.T @ hat)
    
    dofest = 0
    for i in range(X.shape[0]):
        pred, hatvector = NSMap(X, Y, tx, X[i], tx[i], theta, delta, return_hat=True)
        dofest += hatvector[i]
    return dofest

# leaves one input and output pair out, and use rest as training data
# returns predictions which are the length of the whole time series
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

def logLikelihood(X, Y, tx, theta, delta, returnSeries=False):
    
    n = Y.shape[0]

    Yhat = leaveOneOut(X, Y, tx, theta, delta)
    
    # mean_squared_residuals = np.sum((Y-Yhat)**2) / n
    
    ### VERSION WITH MODEL DEGREES OF FREEDOM INCORPORATED
    k = dofestimation(X, Y, tx, theta, delta)
    print(f"dof = {k}")
    mean_squared_residuals = np.sum((Y-Yhat)**2) / (n-k)

    lnL = (-n/2)*(np.log(mean_squared_residuals) + np.log(2*np.pi) + 1 )

    if returnSeries:
        return (lnL, Yhat)
    else:
        return lnL

# make a 1 time step prediction based on a given state(nD vector)
def SMap(X, Y, x, theta):
    norms = la.norm(X-x,axis=1)
    d = np.mean(norms) # d = np.mean(norms) # 
    
    W = np.diag(np.exp(-1 * theta * norms / d))

    H = getHat(X, W, x)
    return H @ Y

### TIME IS NOT INCLUDED AS A STATE VARIABLE ###
# INPUTS
#   X - (ndarray) training data, (n,p) array of state space variables
#   Y - (ndarray) labels
#   T - (ndarray) time for each row in X
#   x - (ndarray) current state to predict from
#   t - (scalar) current time to predict from
#   theta - (scalar) hyperparameter
#   delta - (scalar) hyperparameter
# Note that T and t(where) must be standardized to be between 0 and 1 

def NSMap(X, Y, T, x, t, theta, delta, return_hat=False, return_hat_derivatives=False):
    # create weights

    n = X.shape[0]

    norms = la.norm(X - x,axis=1)
    d = np.mean(norms)

    W = np.exp(-1*(theta*norms)/d - delta*(T-t)**2)[:,None]
    M = np.hstack([X, np.ones((n,1))])
    xaug = np.hstack([x, 1]).T

    if return_hat or return_hat_derivatives:
        pinv = la.pinv(W*M)

        H = xaug @ (pinv.T * W).T
        prediction = (H @ Y)[0]

        if return_hat_derivatives:
            dWdtheta = -1 * W.flatten() * norms / d
            dWddelta = -1 * W.flatten() * ((T-t)**2)

            dthetapinv = (dWdtheta[:,None].T * pinv)
            ddeltapinv = (dWddelta[:,None].T * pinv)

            dhdtheta = 2 * xaug @ (dthetapinv - dthetapinv @ M @ (pinv * W.T))
            dhddelta = 2 * xaug @ (ddeltapinv - ddeltapinv @ M @ (pinv * W.T))

            return (prediction, H, dhdtheta, dhddelta)
    
        return (prediction, H)
    else:
        prediction = xaug @ la.lstsq( W * M, W * Y, rcond=None)[0]
        return prediction

def get_delta_agg(Xr, maxLags, t=None, horizon=1, tau=1, trainingSteps=100, return_forecast_skill=False, theta_fixed=False, make_plots=False):
    
    if t is None:
        t = np.linspace(0,1, num=len(Xr))
    else:
        # Remember to standardize t to be between 0 and 1!
        assert t[0] == 0 and t[-1] == 1

    table = np.zeros((maxLags+1, 5))
    hp = np.zeros(2)

    # produce delay embedding vector first so the set of targets is fixed across all E
    Xemb, Y, tx = delayEmbed(Xr, horizon, maxLags, tau, t=t)

    # for each number of lags from 0 to maxLags
    for l in range(maxLags+1):
        X = Xemb[:,:l+1]

        # print("NSMap")
        thetaNS, deltaNS, lnLNS = optimizeG(X, Y, tx, fixed=np.array([theta_fixed, False]), trainingSteps=trainingSteps, hp=hp.copy())
        # print("SMap")
        thetaS, _, lnLS = optimizeG(X, Y, tx, fixed=np.array([theta_fixed, True]),trainingSteps=trainingSteps, hp=hp.copy())

        table[l] = np.array([deltaNS, lnLNS, lnLS, thetaNS, thetaS])

    if make_plots:
        make_delta_plots(Xr, t, maxLags, table)

    lnLdifference = table[:,1] - table[:,2]
    # ns_area =  np.sum(np.maximum(lnLdifference, np.zeros(maxLags+1)))
    delta_agg_weights = np.exp(lnLdifference - np.max(lnLdifference))
    delta_agg = np.average(table[:,0], weights=delta_agg_weights)
    theta = table[np.argsort(table[:,1])[-1],3]

    if return_forecast_skill:
        return (delta_agg, theta, get_r_sqrd(table, Xemb, Y, tau, tx))
    else: 
        return delta_agg

def make_delta_plots(Xr, t, maxLags, table):
    fig, ax = plt.subplots(1)

    fsize = 25
    E_range = range(1,maxLags+2)

    ax.plot(E_range, table[:,0],label=r"$\hat{\delta}$")
    ax.set_xlabel("E", size = fsize)
    ax.set_ylabel(r"$\hat{\delta}$", size = fsize, rotation=0)
    ax.set_xticks(E_range)
    ax.tick_params(axis='both', which='major', labelsize=fsize * 3 / 4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='100%',pad=0.1)

    minLine = (table[:,2] * 0)+min(min(table[:,2]),min(table[:,1])) 

    cax.plot(E_range, table[:,2], "r--", label="SMap")
    cax.plot(E_range, table[:,1], "y--", label="NSMap")
    cax.fill_between(E_range, table[:,2], minLine, alpha=0.5, color="red")
    cax.fill_between(E_range, table[:,1], minLine, alpha=0.5, color = "yellow")
    cax.set_xlabel("E", size = fsize)
    cax.set_ylabel(r"$\ln\mathcal{L}$", size = fsize, rotation=0)
    cax.set_xticks(E_range)
    cax.legend(fontsize = fsize)
    cax.legend(fontsize = fsize)
    cax.tick_params(axis='both', which='major', labelsize=fsize * 3 / 4)
    cax.tick_params(axis='both', which='major', labelsize=fsize * 3 / 4)

    plt.tight_layout()
    plt.show()

# ugly but necessary function, finds the r squared coefficient based on the other data from get_delta_agg
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
        # create the train and test stuff
        
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

# Optimize using GRADIENT DESCENT instead of evaluating a grid
def optimizeG(X, Y, t, trainingSteps=40, hp=np.array([0.0,0.0]), fixed=np.array([False, False])):    
    err = 0
    
    gradPrev = np.ones(hp.shape, dtype=float)
    deltaPrev = np.ones(hp.shape, dtype=float)
    
    for count in range(trainingSteps):
        errPrev = err
        grad, err = gradient(X, Y, t, hp[0], hp[1])

        # print(f"[{count+1:02d}] theta: {hp[0]:.3f}, delta: {hp[1]:.3f}, log Likelihood: {err:.3f}")

        if abs(err-errPrev) < 0.01 or count == trainingSteps-1:
            break

        dweights, deltaPrev, gradPrev = calculateHPChange(grad, gradPrev, deltaPrev)
         
        # floor and ceiling on the hyperparameters
        for i in range(2):
            if not fixed[i]:
                hp[i] = max(0, hp[i] + dweights[i])

    return (hp[0], hp[1], err)

def calculateHPChange(grad, gradPrev, deltaPrev):
    rhoplus = 1.2 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1
    
    grad = grad / la.norm(grad)# np.abs(grad) # NORMALIZE, because rprop ignores magnitude

    s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
    spos = np.ceil(s) # 0 for - vals, 1 for + vals
    sneg = -1 * (spos - 1)

    delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
    dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

    return (dweights, delta, grad)
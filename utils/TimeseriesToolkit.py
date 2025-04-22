import numpy as np
import numpy.linalg as la

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
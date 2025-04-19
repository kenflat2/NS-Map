import NSMapToolkit as ns

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import numpy.random as rand
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

# generate a sinusoidal time series with mean 0, standard deviation 1,
# and 10% observation noise
def generateLinearSeries(length=200, obs_noise=0.1, theta=lambda t: np.pi/6):
    init = rand.random(1) * 2 * np.pi
    
    t = np.linspace(0, 1, num = length)
    ts = np.sin(t * length * theta(t) + init) * np.sqrt(2)
    ts = ts + rand.normal(0, obs_noise, length)

    return ts

ts = ns.standardize(generateLinearSeries())
X, Y, tx = ns.delayEmbed(ts, 1, 1, np.linspace(0,1,num=len(ts)))

θ, δ, err = ns.optimizeG(X, Y, tx)

assert((θ < 10e-2) and (δ < 10e-2))

X, Y, tx = ns.delayEmbed(ts, 0, 1, np.linspace(0,1,num=len(ts)))
E0_dof = ns.dofestimation(X, Y, tx, 0, 0)

X, Y, tx = ns.delayEmbed(ts, 1, 1, np.linspace(0,1,num=len(ts)))
E1_dof = ns.dofestimation(X, Y, tx, 0, 0)

assert(E0_dof < E1_dof)
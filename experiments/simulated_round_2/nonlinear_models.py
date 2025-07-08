import os
import sys

# Hard code the root directory
root = '/home/kenflat2/NS-Map/' # os.path.dirname(os.path.dirname(experiment_directory))  # Parent directory of the current file
sys.path.append(root)
# experiment_directory = os.path.join(ROOT_DIR)

import json
import numpy as np
import numpy.random as rand
from pathlib import Path

from utils.TimeseriesToolkit import standardize
from scipy.integrate import odeint

experiment_directory = os.path.join(root, "experiments", "simulated_round_2")

with open(os.path.join(experiment_directory, "parameters_round2.json"), "r") as f:
    params = json.load(f)

def generate_logistic(tlen, r_base, r_slope, observation_noise, process_noise):
    model_params = params["experiments"][0]["parameters"]

    r = lambda t: r_base + r_slope * t

    ts = np.zeros(tlen)
    ts[0] = rand.uniform(0, 1)

    for i in range(tlen-1):
        t = i / (tlen - 1)
        x = r(t) * ts[i] * (1 - ts[i])
        u = np.log(x / (1 - x))
        z = rand.normal(0, process_noise)
        ts[i+1] = 1 / (1 + np.exp(z - u))

    return ts[:,None] + rand.normal(0, observation_noise, tlen)[:, None]

def FoodChainP(xi, t, b1):
    (x,y,z)=xi

    a1 = 5
    a2 = 0.1
    b1 = b1(t)
    b2 = 2
    d1 = 0.4
    d2 = 0.01

    dx = x*(1-x)- a1*x*y/(1+b1*x)
    dy = a1*x*y/(1 + b1*x) - d1*y - a2*y*z/(1 + b2*y)
    dz = a2*y*z/(1 + b2*y) - d2*z

    return dx, dy, dz

def generate_food_chain(tlen, b1_base, b1_trend, observation_noise, process_noise):
    model_params = params["experiments"][1]["parameters"]

    settlingTime = model_params["settling_time"]
    end = model_params["time_per_step"] * tlen
    reduction = model_params["reduction"]

    b1 = lambda t: b1_base + b1_trend * t / end

    x0 = np.ones(3)

    if settlingTime > 0:
        tSettle = np.arange(0,settlingTime, step=end/(reduction*tlen))
        fixed_driver = lambda t: b1_base

        x0 = odeint(FoodChainP, x0, tSettle, args=(fixed_driver,))[-1]
    
    t = np.linspace(0,end,num=tlen*reduction)
    ts = np.zeros((tlen, len(x0)))
    ts[0] = x0

    for i in range(tlen-1):
        ts[i+1] = odeint(FoodChainP, ts[i], t[i*reduction:(i+1)*reduction], args=(b1,))[-1] * np.exp(rand.normal(0,process_noise))

    return ts[:, 0, None] + rand.normal(0, observation_noise, (tlen, 1))
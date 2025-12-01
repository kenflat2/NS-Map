import os
import sys

# Hard code the root directory
root = '/net/flood/home/kengee/NS-Map/NS-Map' # '/home/kenflat2/NS-Map/' # os.path.dirname(os.path.dirname(experiment_directory))  # Parent directory of the current file
sys.path.append(root)

"""
# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
experiment_directory = os.path.join(root, "experiments", "simulated_round_1")
sys.path.append(root)
"""

import json
import numpy as np
import numpy.random as rand
from pathlib import Path

from utils.TimeseriesToolkit import standardize
from scipy.integrate import odeint

experiment_directory = os.path.join(root, "experiments", "simulated_round_1")

# experiment_directory = "/experiments/simulated_round_1/"

# print(root + "/parameters_linear.json")

with open(experiment_directory + "/parameters_round1.json", "r") as f:
    params = json.load(f)

## MODELS TO BE TESTED ##

def generate_linear():
    model_params = params["experiments"][0]["parameters"]

    time_series_length = model_params["time_series_length"]

    theta = lambda t: 2 * np.pi * t / model_params["period"]
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(time_series_length)) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], time_series_length)
    return (time_series + obs_noise)[:, None]

def generate_linear_nonstat():
    model_params = params["experiments"][1]["parameters"]
    time_series_length = model_params["time_series_length"]

    theta = lambda t: 2 * np.pi / model_params["period"] * ((t / model_params["period"]) ** 2)
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(time_series_length)) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], time_series_length)
    return (time_series + obs_noise)[:, None]

def generate_logistic():
    model_params = params["experiments"][2]["parameters"]

    r = model_params["param_base"]
    observation_noise = model_params["obs_noise"]
    process_noise = model_params["process_noise"]
    tlen = model_params["time_series_length"]

    ts = np.zeros(tlen)
    ts[0] = rand.uniform(0, 1)

    for i in range(tlen-1):
        t = i / (tlen - 1)
        x = r * ts[i] * (1 - ts[i])
        u = np.log(x / (1 - x))
        z = rand.normal(0, process_noise)
        ts[i+1] = 1 / (1 + np.exp(z - u))

    return standardize(ts[:,None]) + rand.normal(0, observation_noise, tlen)[:, None]

def generate_logistic_nonstat():
    model_params = params["experiments"][3]["parameters"]

    r_base = model_params["nonstat_param_base"]
    r_slope = model_params["nonstat_param_slope"]
    observation_noise = model_params["obs_noise"]
    process_noise = model_params["process_noise"]
    tlen = model_params["time_series_length"]

    r = lambda t: r_base + r_slope * t

    ts = np.zeros(tlen)
    ts[0] = rand.uniform(0, 1)

    for i in range(tlen-1):
        t = i / (tlen - 1)
        x = r(t) * ts[i] * (1 - ts[i])
        u = np.log(x / (1 - x))
        z = rand.normal(0, process_noise)
        ts[i+1] = 1 / (1 + np.exp(z - u))

    return standardize(ts[:,None]) + rand.normal(0, observation_noise, tlen)[:, None]

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

def generate_food_chain():
    model_params = params["experiments"][4]["parameters"]

    tlen = model_params["time_series_length"]
    b1_base = model_params["nonstat_param"]
    observation_noise = model_params["obs_noise"]
    process_noise = model_params['process_noise']

    settlingTime = model_params["settling_time"]
    end = model_params["time_per_step"] * tlen
    reduction = model_params["reduction"]

    b1 = lambda t: b1_base

    x0 = np.random.uniform(0, 1, 3)

    if settlingTime > 0:
        tSettle = np.arange(0,settlingTime, step=end/(reduction*tlen))
        fixed_driver = lambda t: b1_base

        x0 = odeint(FoodChainP, x0, tSettle, args=(fixed_driver,))[-1]
    
    t = np.linspace(0,end,num=tlen*reduction)
    ts = np.zeros((tlen, len(x0)))
    ts[0] = x0

    for i in range(tlen-1):
        ts[i+1] = odeint(FoodChainP, ts[i], t[i*reduction:(i+1)*reduction], args=(b1,))[-1] # * np.exp(rand.normal(0,process_noise))
    
    return standardize(ts[:, 0, None]) + rand.normal(0, observation_noise, (tlen, 1))
    # return ts + rand.normal(0, observation_noise, (tlen, len(x0)))

def generate_food_chain_nonstat():
    model_params = params["experiments"][5]["parameters"]

    tlen = model_params["time_series_length"]
    b1_base = model_params["nonstat_param_base"]
    b1_trend = model_params["nonstat_param_slope"]
    observation_noise = model_params["obs_noise"]
    process_noise = model_params['process_noise']

    settlingTime = model_params["settling_time"]
    end = model_params["time_per_step"] * tlen
    reduction = model_params["reduction"]

    b1 = lambda t: b1_base + (tlen / tlen) * b1_trend * t / end

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

    return standardize(ts[:, 0, None]) + rand.normal(0, observation_noise, (tlen, 1))
    # return ts + rand.normal(0, observation_noise, (tlen, len(x0)))
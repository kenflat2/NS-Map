import json
import numpy as np
import os
import numpy.random as rand

import sys

sys.path.append("../../")
from utils.TimeseriesToolkit import standardize

with open("parameters_linear.json", "r") as f:
    params = json.load(f)

## MODELS TO BE TESTED ##
def generate_stationary_linear():
    model_params = params["experiments"][0]["parameters"]

    theta = lambda t: 2 * np.pi / model_params["period"] * t
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(params["length"])) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], params["length"])
    return time_series + obs_noise

def generate_nonstationary_trend_linear():
    model_params = params["experiments"][1]["parameters"]

    theta = lambda t: 2 * np.pi / model_params["period"] * t
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(params["length"])) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], params["length"])
    trend = model_params["trend"] * np.linspace(0,1,num=params["length"])
    return time_series + obs_noise + trend

def generate_nonstationary_variance_increase_linear():
    model_params = params["experiments"][2]["parameters"]

    theta = lambda t: 2 * np.pi / model_params["period"] * t
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(params["length"])) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], params["length"])
    variance_increase = model_params["variance_increase"] * np.linspace(1, 2, num = params["length"])
    return time_series * variance_increase + obs_noise

def generate_nonstationary_oscillation_speed_linear():
    model_params = params["experiments"][3]["parameters"]

    theta = lambda t: 2 * np.pi / model_params["period"] * ((t / model_params["period"]) ** 2)
    x0 = rand.random(1)[0] * 2 * np.pi
    time_series = np.sin(theta(np.arange(params["length"])) + x0) * np.sqrt(2)
    obs_noise = rand.normal(0, model_params["obs_noise"], params["length"])
    return time_series + obs_noise

def generate_ricker_series(k, mu=0.0):

    x0 = k(0)

    ts = np.zeros(params["length"])
    ts[0] = x0
    ricker = lambda x, t: x * np.exp(1 - x / k(t)) + mu * rand.normal(0, 1)

    for i in range(1, len(ts)):
        ts[i] = ricker(ts[i-1], (i-1)/(params["length"]-1))
    
    return standardize(ts)

def generate_stationary_equilibrium():

    model_params = params["experiments"][4]["parameters"]

    k = lambda t: 1.0

    time_series = generate_ricker_series(k, mu=model_params["obs_noise"])
    # obs_noise = rand.normal(0, model_params["obs_noise"], params["length"])
    return time_series # + obs_noise

def generate_nonstationary_equilibrium_trend():
    model_params = params["experiments"][5]["parameters"]

    k = lambda t: 1.0 + model_params["trend"] * t
    time_series = generate_ricker_series(k, mu=model_params["obs_noise"])

    return time_series # + obs_noise + trend
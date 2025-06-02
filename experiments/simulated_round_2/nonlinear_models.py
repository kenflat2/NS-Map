import os
import sys

# Dynamically set the root directory
root = os.path.dirname(os.path.abspath(__name__))  # Current file's directory
sys.path.append(root)
# experiment_directory = os.path.join(ROOT_DIR)

import json
import numpy as np
import numpy.random as rand
from pathlib import Path

from utils.TimeseriesToolkit import standardize


experiment_directory = "/experiments/linear/"

print(root + experiment_directory + "parameters_linear.json")

with open(root + experiment_directory + "parameters_linear.json", "r") as f:
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

def generate_food_chain(tlen, b1_base, b1_trend, observation_noise, process_noise):
    model_params = params["experiments"][1]["parameters"]


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
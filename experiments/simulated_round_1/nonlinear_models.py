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
    return time_series

def generate_nonstationary_equilibrium_trend():
    model_params = params["experiments"][5]["parameters"]

    k = lambda t: 1.0 + model_params["trend"] * t
    time_series = generate_ricker_series(k, mu=model_params["obs_noise"])

    return time_series

def generate_autoregressive():
    model_params = params["experiments"][6]["parameters"]

    yii=np.zeros(params["length"]); yii[0]=10
    a=model_params["a"]
    for i in range(1,params["length"]):
        yii[i]=yii[i-1]*a+np.random.normal(0, model_params["process_noise"])

    return yii

def generate_autoregressive_plus_sinusoid():
    model_params = params["experiments"][7]["parameters"]

    yii=np.zeros(params["length"]); yii[0]=10
    a=model_params["a"]
    for i in range(1,params["length"]):
        yii[i]=yii[i-1]*a+np.random.normal(0, model_params["process_noise"])

    t = np.arange(params["length"])
    y_sin = np.sin(2*np.pi/12*t)+np.random.normal(0, model_params["obs_noise"], params["length"])
    y = y_sin + yii

    return y

def generate_autoregressive_plus_sinusoid2():
    model_params = params["experiments"][8]["parameters"]

    yii=np.zeros(params["length"]); yii[0]=10
    a=model_params["a"]
    for i in range(1,params["length"]):
        yii[i]=yii[i-1]*a+np.random.normal(0, model_params["process_noise"])

    t = np.arange(params["length"])
    y_sin = np.sin(2*np.pi/12*t)+np.random.normal(0, model_params["obs_noise"], params["length"])
    y = y_sin + 0.3*yii

    return y
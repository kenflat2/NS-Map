import json
import numpy as np
import os
import numpy.random as rand

with open("parameters_linear.json", "r") as f:
    params = json.load(f)

## MODELS TO BE TESTED ##
def generate_stationary():
    theta = lambda t: 2 * np.pi / params["period"] * t
    x0 = rand.random(1)[0] * 2 * np.pi
    return np.sin(theta(np.arange(params["length"])) + x0) * np.sqrt(2) + rand.normal(0, params["obs_noise"], params["length"])
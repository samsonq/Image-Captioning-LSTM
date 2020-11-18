################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from experiment import Experiment
import sys
import os
import numpy as np
import torch
from file_utils import read_file

config = read_file("./default.json")

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
    extras = {"num_workers": config["dataset"]["num_workers"], "pin_memory": True}
else:
    device = torch.device("cpu")
    extras = False

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    exp.test()

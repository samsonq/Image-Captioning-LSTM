import os
import numpy as np
import torch

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
else:
    device = torch.device("cpu")
    extras = False


def load_data():
    """

    :return:
    """
    return

#Optimizing Model Parameters

#Training a model is an iterative process; in each iteration the model makes a guess about 
# the ouput, calculates the error in its guess(loss), collects the derivates of the error 
# with respect to its parameters, and optimizes these parameters using gradient descent.

#Previous code from datasets.py and model.py:

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

##Get Device for Training

#We want to train our model on a hardware accelerator like the GPU or MPS. 
# Check if available, otherwise use the CPU.

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

##Define the Class

#We define our neural network by subclassing nn.Module, and initialize the 
# neural network layers in __init__. Every nn.Module subclass implements 
# the operations on input data in the forward method.

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#Create an instance of NeuralNetwork, move it to the device, and print its structure.
model  = NeuralNetwork().to(device)
print(model)
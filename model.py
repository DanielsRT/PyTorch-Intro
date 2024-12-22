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
print(f"Using {device} device\n")

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

#To use the model, pass it the input data. This executes the model's forward, along 
# with some background operations. Do not call model.forward(directly).

#Calling the model on the input returns a 2-d tensor with dim=0 corresponding to each 
# output of 10 raw predicted values for each class, and dim=1 corresponding to the 
# individual values of each output. We get the prediction probabilities by passing it 
# through an instance of the nn.Softmax module.

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}\n")

##Model layers

#Take a sample minibatch of 3 images of size 28x28 and see what happens to it as we 
# pass it through the network.

input_image = torch.rand(3,28,28)
print(input_image.size())

#Initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous 
# array of 784 pixel values(the minibatch dimension (at dim=0) is maintained).

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#The linear layer is a module that applies a linear transformation on the input using 
# its stored weights and biases.

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#Non-linear activations are what create the complex mappings between the model's inputs 
# and outputs. They are applied after linear transformations to introduce nonlinearity, 
# helping neural networks learn a wide variety of phenomena.

#In this model, we use nn.ReLU between our linear layers, but there's other activations 
# to introduce non-linearity in your model.

print(f"\nBefore RELU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
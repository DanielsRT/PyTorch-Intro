##Automatic Differentiation with torch.autograd

#When training neural networks, the most frequently used algorithm is back 
# propagation. In this algorithm, model weights are adjusted according to 
# the gradient of the loss function with respect to the given parameter.

#To compute those gradients, PyTorch has a built-in differentiation engine 
# called torch.autograd. It supports automatic computation of gradient for 
# any computational graph.

#Consider the simplest one-layer neural network with input x, parameters w 
# and b, and som eloss function. It can be defined in PyTorch in the following 
# manner:

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#In this network, w and b are parameters which we need to optimize. To compute the 
# gradients of loss function with respect to those variables, set the requires_grad 
# property of those tensors.

#You can set the value of requires_grad when created a tensor, or later by using x.requires_grad_(True)

#This object knows how to compute the function in the forward direction, and also how 
# to compute its derivative during the backward propagation step. A reference to the 
# backward propagation function is stored in grad_fn property of a tensor.

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
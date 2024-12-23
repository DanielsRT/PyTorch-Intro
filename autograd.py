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

##Computing Gradients

#To optimize weights of parameters in the neural network, we need to compute the derivatives 
# of our loss function with respect to parameters, namely we need ∂loss/∂w and ∂loss/∂b under 
# some fixed values of x and y. To compute those derivates, we call loss.backward(), and then 
# retireve the values from w.grad and b.grad:

loss.backward()
print(w.grad)
print(b.grad)

##Disable Gradient Tracking

#By default, all tensors with requires_grad=True are tracking their computational history and 
# support gradient computation. However, there are some cases when we do not need to do that, 
# for example, when we have trained the model and just want to apply it to some input data, i.e.
#  we only want to do forward computations through the network. We can stop tracking computations 
# by surrounding our computation code with torch.no_grad() block:

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

#Another way to achieve the same result is to use the detach() method on the tensor:

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

#In many cases, we have a scalar loss function, and we need to compute the gradient with 
# respect to some parameters. However, there are cases when the output function is an 
# arbitrary tensor. In this case, PyTorch allows you to compute so-called Jacobian product, 
# and not the actual gradient.

#Instead of computing the Jacobian matrix itself, PyTorch allows you to compute Jacobian 
# Product for a given input vector v = (v1 ... vm). This is achieved by calling backward 
# with as v an argument. The size of v should be the same as the size of the original tensor, 
# with respect to which we want to compute the product:

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
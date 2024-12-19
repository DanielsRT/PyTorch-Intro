import torch
import numpy as np

## Initializing Tensors

#Tensors can be created directly from data. Data type is inferred
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

#Tensors can be created from NumPy arrays and vice versa
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#Tensors created from another tensor retains the properties unless overwritten
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# 'shape' is a tupe of tensor dimensions. It can be used as an argument to set the dimensions of an output tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


## Tensor Attributes

#Tensor attributes describe their shape, datatype, and the device on which they are stored
tensor = torch.rand(3,4)

print(f"\nShape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


## Tensor Operations

#Operations can be run on the GPU, typically at higher speeds. Tensors are created on CPU by default. 
# Move tensors to GPU using '.to' if GPU is available. Copying large tensors across devices can be 
# expensive in terms of time and memory.

#Move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#numpy-like indexing and slicing
tensor = torch.ones(4,4)
print(f"\nFirst row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
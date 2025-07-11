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
print(f"Last column: {tensor[..., -1]}\n")
tensor[:,1] = 0
print(tensor)

#Use torch.cat to concatenate a sequence of tensors along a given dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"{t1}\n")

#Compute the matrix multiplication between two tensors. y1, y2, y3 will have the same value
#'tensor.T' returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)

#Compute the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#If you have a one-element tensor, for example by aggregating all values of a tensor into one value,
# you can convert it to a Python numerical value using 'item()'
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#Operations that store the result into the operand and called in-place. They are denoted by a '_' suffix. 
# For example: 'x.copy_(y)' will change x.

#In-place operations save some memory, but can be problematic when computing derivatives because of an 
# immediate loss of history; Hence, their use is discouraged.
print(f"{tensor} \n")
tensor.add_(5)
print(f"{tensor} \n")


## Bridge with NumPy

#Tensors on the CPU and NumPy arrays can share underlying memory locations, and changing one will change the other.
#Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#A change in the tensor reflects in the NumPy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}\n")

#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

#A change in the NumPy array reflects in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
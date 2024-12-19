import torch
import numpy as np

#Initializing tensors

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
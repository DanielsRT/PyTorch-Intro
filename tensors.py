import torch
import numpy as np

#Initializing tensors

#Tensors can be created directly from data. Data type is inferred
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


#Tensors can be created from NumPy arrays and vice versa
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

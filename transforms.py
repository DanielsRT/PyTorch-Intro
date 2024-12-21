#Use transforms to manipulate the data and make it suitable for training. 
# All TorchVision datasets have two parameters - 'transform' to modify the 
# features and 'target_transform' to modify the labels- that accept callables 
# containing the transformation logic.

#The FashionMNIST features are in PIL Image format, and the labels are integers. 
# For training, we need the features as normalized tensors, and the labels as 
# one-hot encoded tensors. To make these transformations, use 'ToTensor' and 'Lambda'

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float)
        .scatter_(0, torch.tensor(y), value=1))
)

#ToTensor converts a PIL image or NumPy ndarray into a FloatTensor and scales the image's 
# pixel intensity values in the range[0., 1.]

#Lambda transforms apply any user-defined lambda function. Above, we define a function to 
# turn the integer into a one-hot encoded tensor. If first creates a zero tensor of size 
# 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on 
# the index as given by label y.
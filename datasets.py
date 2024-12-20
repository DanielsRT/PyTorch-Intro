#Load the 'FashionMNIST' dataset 
# root is the path where the train/test data is stored, 
# train specifies training or test dataset, 
# download=True downloads data from the internet if it isn't available at root, 
# transform and target_transform specify the feature and label transformations

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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
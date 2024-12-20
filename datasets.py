import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#Load the 'FashionMNIST' dataset 
# root is the path where the train/test data is stored, 
# train specifies training or test dataset, 
# download=True downloads data from the internet if it isn't available at root, 
# transform and target_transform specify the feature and label transformations

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

##Iterating and Visualizing the Dataset

#You can index Datasets manualy like a list: 'training_data[index]'. We can use 'matplotlib' 
# to visualize some samples in our training data.

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

##Create a Custom Dataset Class

#Custom 'Dataset' classes must implement '__init__', '__len__', and '__getite__'. In this implementation, 
# FashionMNST images are stored in 'img_dir', and their labels are stored separately in a CSV file 'annotations_file'

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    '''run once when instantiating the Dataset object.'''
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    '''returns the number of samples in the dataset'''
    def __len__(self):
        return len(self.img_labels)
    
    '''
    returns a sample from the dataset at the given index 'idx'. Based on the index, it find's the image's location 
    on disk and converts that to a tensor using 'read_image'. Retrieves the corresponding label in 'self.img_labels' 
    and calls the transform functions on them (if applicable). returns the tensor image and corresponding label in a tuple.
    '''
    def __getItem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

##Prepare data for training with DataLoaders

#The Dataset retrieves features and lables one sample at a time. while training a model, you typically want to pass samples in batches, 
# shuffle the data at every epoch to reduce model overfitting, and use Python's multiprocessing to speed up data retrieval.

#DataLoader is an iterable that abstracts this complexity in an easy API

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
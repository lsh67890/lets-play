import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

seed = 1
lr = 0.001
momentum = 0.5
batch_size = 64
test_batch_size = 64
epochs = 1
no_cuda = False
log_interval = 100

train_paths = glob('dataset/mnist_png/training/*/*.png')
test_paths = glob('dataset/mnist_png/testing/*/*.png')

class Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        # read image
        image = Image.open(path).convert('L')
        # get label
        label = int(path.split('₩₩')[-2])

        if self.transform:
            image = self.transform(image)

        return image, label

data_loader = torch.utils.data.DataLoader(
    Dataset(train_paths,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406],
                    std=[0.225]
                )
            ])
    ),
    batch_size=batch_size,
    shuffle=True
)

path = 'tvxq.jpg'
img = Image.open(path)

# # cropping an image
# torchvision.transforms.CenterCrop(size=(300, 300))(img)
#
# # adjusting colors
# torchvision.transforms.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0)(img)
#
# # cropping image by five parts
# torchvision.transforms.FiveCrop(size=(75, 75))(img)
#
# # converting image to grayscale
# torchvision.transforms.Grayscale(num_output_channels=1)(img)
#
# # adding padding around the image
# torchvision.transforms.Pad(padding=(20, 20), fill=0, padding_mode='constant')(img)
#
# # rotating/affining image
# torchvision.transforms.RandomAffine(degrees=90, fillcolor=0)(img)
#
# transforms = [torchvision.transforms.Grayscale(num_output_channels=1),
#               torchvision.transforms.CenterCrop(size=(300, 300)),
#               torchvision.transforms.RandomAffine(degrees=90, fillcolor=0)]
#
# # choosing a random function from transforms
# torchvision.transforms.RandomChoice(transforms)(img)
#
# # randomly cropping image
# torchvision.transforms.RandomCrop(size=(300, 300))(img)
#
# # randomly convert to grayscale
# torchvision.transforms.RandomGrayscale(p=0.5)(img)
#
# # randomly rotate by horizontal
# torchvision.transforms.RandomHorizontalFlip(p=0.5)(img)
#
# # resizing image (can remove interpolation by default)
# torchvision.transforms.Resize((100, 100), interpolation=2)(img)

tensor = torchvision.transforms.ToTensor()(img)
# mean value is for the dimension and std is for standard deviation
trans = torchvision.transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))(tensor)
print(np.min(img), np.max(img))
print(np.min(trans.numpy()), np.max(trans.numpy())) # value is changed after normalization

# clipping/erasing a certain area of image
trans = torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)(tensor)
plt.imshow(trans.numpy().transpose(1, 2, 0))
plt.show()
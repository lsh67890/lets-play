import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

batch_size = 32
test_batch_size = 32

# 1. data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                   ])),
    batch_size=batch_size,
    shuffle=True
)

# 2. test loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5))
                   ])),
    batch_size=batch_size,
    shuffle=True
)

# 3. data check
imgs, labels = next(iter(train_loader))
print(3.1, imgs.shape)
print(3.2, labels.shape)

# 4. data visualisation
print(4.1, imgs[0].shape)
torch_img = torch.squeeze(imgs[0])
print(4.2, torch_img.shape)
img = torch_img.numpy()
print(4.3, img.shape)
label = labels[0].numpy()
print(4.4, label.shape)
plt.title(label)
plt.imshow(img, 'gray')
plt.show()
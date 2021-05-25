import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 1. Load mnist train data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1
)
img, label = next(iter(train_loader))
print(img.shape, label.shape)
plt.imshow(img[0, 0, :, :], 'gray')
plt.show()

# 2. Build network
layer = nn.Conv2d(1, 20, 5, 1).to(torch.device('cpu')) # first layer needs to be convoluted
print(layer)
weight = layer.weight
print(weight) # corresponds to the layer
weight = weight.detach().numpy()
plt.imshow(weight[0, 0, :, :], 'jet')
plt.show()
output_data = layer(img)
output_data = output_data.data
output = output_data.cpu().numpy()
print(output.shape)
img_arr = img.numpy()
plt.figure(figsize=(15, 30))
plt.subplot(131)

plt.title('Input')
plt.imshow(np.squeeze(img_arr), 'gray')
plt.subplot(132)
plt.title('Weight')
plt.imshow(weight[0, 0, :, :], 'jet')
plt.subplot(133)
plt.title('Output')
plt.imshow(weight[0, 0, :, :], 'gray')
plt.show()

# 3. pooling
pool = F.max_pool2d(img, 2, 2)
pool_arr = pool.numpy()
plt.figure(figsize=(10, 15))
plt.subplot(121)
plt.title("input")
plt.imshow(np.squeeze(img_arr), 'gray')
plt.subplot(122)
plt.title('output')
plt.imshow(np.squeeze(pool_arr), 'gray')
plt.show()

# 4. linear
flatten = img.view(1, 28 * 28) # size will be [1, 784
lin = nn.Linear(784, 10)(flatten)
plt.imshow(lin.detach().numpy(), 'jet')
plt.show()

# 5. softmax
with torch.no_grad():
    flatten = img.view(1, 28 * 28)
    lin = nn.Linear(784, 10)(flatten)
    softmax = F.softmax(lin, dim=1)
    # np.sum(softmax.numpy()) will be 1

# building the layers using Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        # feature extraction
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        print(x)
        # fully connected
        x = x.view(self.conv1(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
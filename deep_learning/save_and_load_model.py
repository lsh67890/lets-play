import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

seed = 1
lr = 0.001
momentum = 0.5
batch_size = 64
test_batch_size = 64
epochs = 10
no_cuda = False
log_interval = 100

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# preprocess
train_paths = glob('dataset/mnist_png/training/*/*.png')[:1000]
test_paths = glob('dataset/mnist_png/testing/*/*.png')[:1000]
print(len(train_paths), len(test_paths))

class Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        img = Image.open(path).convert('L')
        label = int(path.split('\\')[-2])
        if self.transform:
            img = self.transform(img)
        return img, label

torch.manual_seed(seed)

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    Dataset(train_paths,
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406],
                    std=[0.225])])
            ),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    Dataset(test_paths,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406],
                    std=[0.225]
                )
            ])),
    batch_size=batch_size,
    shuffle=True
)

for i, data in enumerate(train_loader):
    if i == 0:
        print(data[0].shape, data[1].shape)
        break

# optimization
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# training
for epoch in range(1, epochs+1):
    # train model
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{} / {} ({}:.0f%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # test mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy
    ))

# save weight
save_path = 'model_weight.pt'
torch.save(model.state_dict(), save_path)
print(model.state_dict()) # it includes layers of the model
print(model.state_dict()['conv1.weight'].shape)
weight_dict = torch.load('model_weight.pt')
model.load_state_dict(weight_dict)
model.eval()

# save entire model
# the advantage of saving the entire model is that loading it should be enough
save_path = 'model.pt'
torch.save(model, save_path)
model = torch.load(save_path)
model.eval()

# save, load, resuming training
checkpoint_path = 'checkpoint.pt'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, checkpoint_path)
model = Net().to(device) # at this stage the model is empty
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

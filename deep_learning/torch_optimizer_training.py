import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

seed = 1
batch_size = 64
test_batch_size = 64
no_cuda = False

use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 1. preprocess
torch.manual_seed(seed)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),  (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True
)

# 2. model
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

# 3. optimization
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
params = list(model.parameters())
for i in range(8): # check the parameters (weight, bias))
    print(params[i].size())

# 4. before training
# convert the model to train mode to enable training
# prepare to train with convolution, linear, dropout, batch normalization and layers with parameters
model.train() # set train mode
data, target = next(iter(train_loader)) # extract the first batch data to put into the model
print(data.shape, target.shape)
data, target = data.to(device), target.to(device) # compile the extracted batch data on the device
optimizer.zero_grad() # clear the optimizer to find the best value
output = model(data) # put in the prepared data into the model to get the output (prediction)
loss = F.nll_loss(output, target) # input the predicted result into the loss function (using negative loss likelihood)
loss.backward() # calculate the gradients by back propagation
optimizer.step() # calculated gradients are updated in the parameter

# 5. start training (repeat step 4)
epochs = 1
log_interval = 100
for epoch in range(1, epochs+1):
    model.train() # train model
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clear the gradient
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format( # loss will continuously decrease
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.item()
            ))
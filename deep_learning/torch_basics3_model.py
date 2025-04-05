import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
from torchvision.transforms import ToTensor
from torchvision import datasets
import matplotlib.pyplot as plt

# Load FashionMNIST data
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
print(training_data)

# Visualise data
label_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Create data loader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
# iterate through dataloader and mark image and the answer (label)
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()

# Create simple custom dataset, data loader
class CustomDataset(Dataset):
    def __init__(self, np_data, transform=None):
        self.data = np_data
        self.transform = transform
        self.len = np_data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def square(sample):
    return sample ** 2

trans = tr.Compose([square])
np_data = np.arange(10)
custom_dataset = CustomDataset(np_data, transform=trans)
print(custom_dataset)
custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

# Create Model class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # make 28x28 to 1 row
        self.linear_relu_stack = nn.Sequential( # making it into a sequence means making a layer
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# generate model instance, set up device
model = NeuralNetwork().to(device)

# make virtual data and predict
x = torch.rand(1, 28, 28, device=device)
logits = model(x)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

# define loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# define training / validation(test) function
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# function for testing
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"test error: \n accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f} \n")

# TRAINING PROCESS
epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}\n-------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

# save trained model parameter
torch.save(model.state_dict(), 'model_weights.pth')
# generate new model instance and set device
model2 = NeuralNetwork().to(device)
print(model2)

model2.eval() # eval indicates this model is not for training
test_loop(test_dataloader, model2, loss_fn)

# call saved parameter
model2.load_state_dict(torch.load('model_weights.pth'))

model2.eval()
test_loop(test_dataloader, model2, loss_fn)

# save the entire model and call it
torch.save(model, 'model.pth')
model3 = torch.load('model.pth')
model3.eval()
test_loop(test_dataloader, model3, loss_fn)

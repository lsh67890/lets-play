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
print(f"feature batch shape: {train_features.size()}")
print(f"labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"label: {label}")

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

for _ in range(3):
    for data in custom_dataloader:
        print(data)
    print("="*20)

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
print(model) # check how the layer is structured

# make virtual data and predict
x = torch.rand(1, 28, 28, device=device)
logits = model(x)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"predicted class: {y_pred}")

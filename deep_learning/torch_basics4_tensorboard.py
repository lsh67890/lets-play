import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor

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

writer = SummaryWriter('./logs/pytorch')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

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

# generate new model instance and setup device
model4 = NeuralNetwork().to(device)
print(model4)
model4.eval()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model4.parameters(), lr=learning_rate)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
loss_fn = nn.CrossEntropyLoss()
test_loop(test_dataloader, model4, loss_fn)

x = torch.rand(1, 28, 28, device=device)
writer.add_graph(model4, x) # writes model to tensorboard

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # predict and calculate loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")

        total_loss += loss / len(dataloader)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"test error: \n accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n")
    return test_loss

parameters = ['weight1', 'bias1', 'weight2', 'bias2']

epochs = 10
for t in range(epochs):
    print(f"epoch {t+1}\n------")
    train_loss = train(train_dataloader, model4, loss_fn, optimizer)
    writer.add_scalar('training loss', train_loss, t)
    for param, name in zip(model4.parameters(), parameters):
        writer.add_histogram(name, param, t)
    test_loss = test(test_dataloader, model4, loss_fn)
    writer.add_scalar('test_loss', test_loss, t)
print("done")

writer.close()

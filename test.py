from loading import CustomDatasetFromCSV, csv_from_images_and_labels, split_dataset
from neural_net import run_neural_net
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*(120*120), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


root_dir = 'F:/Users/Henry/Coding/League of Icons/Icons/'
csv_path  = 'F:/Users/Henry/Coding/League of Icons/Icons/icon_labels.csv'
image_dataset = CustomDatasetFromCSV(csv_path, root_dir)

train_labelled_images, dev_labelled_images, test_labelled_images = split_dataset(image_dataset)

train_dataloader = DataLoader(train_labelled_images, batch_size=32)
dev_dataloader = DataLoader(dev_labelled_images, batch_size=32)
test_dataloader = DataLoader(test_labelled_images, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 9 == 8:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = 'F:/Users/Henry/Coding/League of Icons/icon_model.pth'
torch.save(model.state_dict(), PATH)


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images = images.to(torch.float32)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 74 test images: {100 * correct // total} %')

from test import NeuralNetwork
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

root_dir = 'F:/Users/Henry/Coding/League of Icons/Icons/'
csv_path  = 'F:/Users/Henry/Coding/League of Icons/Icons/icon_labels.csv'
image_dataset = CustomDatasetFromCSV(csv_path, root_dir)

train_labelled_images, dev_labelled_images, test_labelled_images = split_dataset(image_dataset)

train_dataloader = DataLoader(train_labelled_images, batch_size=32)
dev_dataloader = DataLoader(dev_labelled_images, batch_size=32)
test_dataloader = DataLoader(test_labelled_images, batch_size=32)

PATH = 'F:/Users/Henry/Coding/League of Icons/icon_model.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

model.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        inputs = inputs.to(torch.float32)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} test images: {100 * correct // total} %'.format(test_dataloader.size()))
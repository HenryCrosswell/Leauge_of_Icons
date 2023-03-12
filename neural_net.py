import torch
from torch import nn
import torch.optim as optim
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*(120*120), 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_neural_network(train_dataloader, lr):
    correct = 0
    total = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum= 0.9)
    incorrect_images_and_label = {}
    for epoch in range(6):  # loop over the dataset multiple times
        running_loss = 0.0
        dataset_length = 0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels, image_name = data
            inputs = inputs.to(torch.float32)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # print statistics
            dataset_length += len(labels)
            for index, actual_labels in enumerate(labels):
                if actual_labels != predicted[index]:
                    current_image = image_name[index]
                    wrong_prediction = int(predicted[index])
                    incorrect_images_and_label[current_image] = wrong_prediction
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 9:.3f}')
                running_loss = 0.0
    print('Finished Training')
    print(f'Accuracy of the network on the {dataset_length} batches of images: {100 * correct // total} %')
    print(f'missclassified images {incorrect_images_and_label.items()}')
    return model

def test_model(dataset, model):
    correct = 0
    total = 0
    dataset_length = 0
    incorrect_images_and_label={}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataset:
            images, dlabels, image_name = data
            images = images.to(torch.float32)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += dlabels.size(0)
            correct += (predicted == dlabels).sum().item()
            dataset_length += len(dlabels)
            for index, actual_labels in enumerate(dlabels):
                if actual_labels != predicted[index]:
                    current_image = image_name[index]
                    wrong_prediction = int(predicted[index])
                    incorrect_images_and_label[current_image] = wrong_prediction
    print(f'Accuracy of the network on the {dataset_length} batches of images: {100 * correct // total} %')
    print(f'missclassified images {incorrect_images_and_label.items()}')
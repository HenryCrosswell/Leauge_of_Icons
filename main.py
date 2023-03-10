from loading import CustomDatasetFromCSV, csv_from_images_and_labels, split_dataset
from neural_net import run_neural_net
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
root_dir = 'F:/Users/Henry/Coding/League of Icons/Icons/'
csv_path  = 'F:/Users/Henry/Coding/League of Icons/Icons/icon_labels.csv'
image_dataset = CustomDatasetFromCSV(csv_path, root_dir)

input_dim = 120*120
hidden_dim = 10000
output_dim = 2


train_labelled_images, dev_labelled_images, test_labelled_images = split_dataset(image_dataset)
print('image_array = {}'.format('train_labelled_images[1][0]'))
print('image label = {}'.format(train_labelled_images[1][1]))

train_dataloader = DataLoader(train_labelled_images, batch_size=32)
dev_dataloader = DataLoader(dev_labelled_images, batch_size=32)
test_dataloader = DataLoader(test_labelled_images, batch_size=32)

train_features, train_labels = next(iter(train_dataloader))
train_features = train_features.to(torch.float32)



run_neural_net(train_features, train_labels)
print('yeah')
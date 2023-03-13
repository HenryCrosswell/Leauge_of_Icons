from dataset_manipulation import CustomDatasetFromCSV, split_dataset, data_split_dataloader
from neural_net import NeuralNetwork, test_model
import torch

root_dir = 'F:/Users/Henry/Coding/League of Icons/Icons/'
csv_path  = 'F:/Users/Henry/Coding/League of Icons/Icons/icon_labels.csv'
model_path = 'F:/Users/Henry/Coding/League of Icons/icon_model.pth'

image_dataset = CustomDatasetFromCSV(csv_path, root_dir)
train_labelled_images, dev_labelled_images, test_labelled_images = split_dataset(image_dataset)

train_dataloader, dev_dataloader, test_dataloader = data_split_dataloader(train_labelled_images, dev_labelled_images, test_labelled_images, 32)

device = torch.device('cpu')
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('F:/Users/Henry/Coding/League of Icons/icon_model.pth'))

test_model(dev_dataloader, model)

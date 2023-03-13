import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path)
        self.root_dir  = root_dir

    def __getitem__(self, index):
        icon_image_name = self.data['icon'][index]
        image_path = os.path.join(self.root_dir, icon_image_name)
        rgba_image = Image.open(image_path)
        image = rgba_image.convert('RGB')
        image = np.array(image)
        label = self.data['label'][index]
        return image, label, icon_image_name

    def __len__(self):
        return len(self.data)

def split_dataset(image_dataset):
    train_split = int(len(image_dataset)*0.7)
    dev_split = int(len(image_dataset)*0.15)
    test_split = int(len(image_dataset)-(dev_split+train_split))
    train_labelled_images, dev_labelled_images, test_labelled_images = torch.utils.data.random_split(image_dataset, [train_split, dev_split, test_split], generator=torch.Generator().manual_seed(42))
    return train_labelled_images, dev_labelled_images, test_labelled_images    

def data_split_dataloader(train_labelled_images, dev_labelled_images, test_labelled_images, batch_size):
    train_dataloader = DataLoader(train_labelled_images, batch_size)
    dev_dataloader = DataLoader(dev_labelled_images, batch_size)
    test_dataloader = DataLoader(test_labelled_images, batch_size)
    return train_dataloader, dev_dataloader, test_dataloader
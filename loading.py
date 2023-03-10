import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
import os 
from skimage import io
#change to df
import pandas as pd 
import os
from os import listdir

def csv_from_images_and_labels(): 
    absolute_path= os.path.dirname(__file__)
    icon_file = "Icons/"
    csv_file = 'icon_labels.csv'
    icons_file_path = os.path.join(absolute_path, icon_file)
    icon_df = pd.DataFrame(columns = ['icon', 'label'])

    for img in listdir(icons_file_path):
        if img.startswith('L'):
            label = 1
        else:
            label = 0
        icon_df.loc[len(icon_df)] = [img, label]
        
    icon_df.to_csv(os.path.join(icons_file_path, csv_file),index=False)

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, root_dir):
        self.data = pd.read_csv(csv_path)
        self.root_dir  = root_dir

    def __getitem__(self, index):
        icon_image_name = self.data['icon'][index]
        image_path = os.path.join(self.root_dir, icon_image_name)
        image = plt.imread(image_path)
        label = self.data['label'][index]
        return image, label

    def __len__(self):
        return len(self.data)



def split_dataset(image_dataset):
    train_split = int(len(image_dataset)*0.8)
    dev_split = int(len(image_dataset)*0.1)
    test_split = int(len(image_dataset)-(dev_split+train_split))
    train_images, dev_images, test_images = torch.utils.data.random_split(image_dataset, [train_split, dev_split, test_split], generator=torch.Generator().manual_seed(2))
    return train_images, dev_images, test_images    
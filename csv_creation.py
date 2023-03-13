
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
        elif img.endswith('.csv'):
            continue
        else:
            label = 0
        icon_df.loc[len(icon_df)] = [img, label]
        
    icon_df.to_csv(os.path.join(icons_file_path, csv_file),index=False)

csv_from_images_and_labels()

from loading import CustomDatasetFromCSV, csv_from_images_and_labels, split_dataset

root_dir = 'F:/Users/Henry/Coding/League of Icons/Icons/'
csv_path  = 'F:/Users/Henry/Coding/League of Icons/Icons/icon_labels.csv'
image_dataset = CustomDatasetFromCSV(csv_path, root_dir)

split_dataset(image_dataset)
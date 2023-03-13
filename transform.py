import os
from os import listdir
from PIL import Image

absolute_path= os.path.dirname(__file__)
icon_file = "Icons/"
icons_file_path = os.path.join(absolute_path, icon_file)

for img in listdir(icons_file_path):
    if img.startswith('L'):
        image = Image.open('F:/Users/Henry/Coding/League of Icons/Icons/'+img)
        box = (8, 8, 112, 112)
        image = image.crop(box)
        image = image.resize((120,120), resample=Image.BICUBIC)
        rgb_im = image.convert("RGB")
        rgb_im.save('F:/Users/Henry/Coding/League of Icons/Unused_images/Lcrop_'+str(img))

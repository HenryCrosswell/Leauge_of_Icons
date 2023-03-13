from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib
from PIL import Image
from pathlib import Path

url = 'https://lol.fandom.com/wiki/Category:Ability_Icons?fileuntil=Dragon+Strike.png#mw-category-media'
url_img_xpath = '//*[@id="mw-category-media"]/ul/li[*]/div/div[1]/div/a/img'


options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',options=options)

driver.get(url)

total_height = int(driver.execute_script("return document.body.scrollHeight"))

for i in range(1, total_height, 5):
    driver.execute_script("window.scrollTo(0, {});".format(i))

imgResults = driver.find_elements(By.XPATH,url_img_xpath)
src = {}

for img in imgResults:
    key = img.get_attribute('alt')
    key = key[:-4]
    value = img.get_attribute('src')
    src[key] = value

for i, key in enumerate(src):
    urllib.request.urlretrieve(str(src[key]),Path(f"./new_unused_images/Others_{key}.jpg"))
    c = Image.open(Path(f"./new_unused_images/Others_{key}.jpg"))
    d = c.resize((120,120), resample=Image.BICUBIC)
    rgb_im = d.convert('RGB')
    rgb_im =  rgb_im.save(Path(f"./new_unused_images/Others_{key}.jpg"))
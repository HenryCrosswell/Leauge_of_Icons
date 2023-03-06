from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib

#url = "https://lol.fandom.com/wiki/Category:Champion_Square_Images"
url = 'https://lol.fandom.com/wiki/Category:Champion_Square_Images?filefrom=YuumiSquare.png#mw-category-media'
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
    if img.get_attribute('width') == '120' and img.get_attribute('height') == '120':
        key = img.get_attribute('alt')
        if 'Circle' in key.split():
            continue
        key = key[:-4]
        value = img.get_attribute('src')
        src[key] = value
    else:
        continue

for i, key in enumerate(src):
    urllib.request.urlretrieve(str(src[key]),"F:/Users/Henry/Coding/League of Icons/Icons/League/League_{}.jpg".format(key))

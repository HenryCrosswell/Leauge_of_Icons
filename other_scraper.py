from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib
from PIL import Image

#url = "https://smite.fandom.com/wiki/Category:God_icons"
#url = 'https://smite.fandom.com/wiki/Category:God_icons?filefrom=T+Ares+GodSlayer+Icon.png#mw-category-media'
#url = 'https://heroesofthestorm.fandom.com/wiki/Hero'
url = 'https://dota2.fandom.com/wiki/Category:Hero_icons'

url_img_xpath = '//*[@id="mw-category-media"]/ul/li[*]/div/div[1]/div/a/img'
#url_img_xpath = '//*[@id="mw-content-text"]/div/table[1]/tbody/tr/td/div/div[*]/a/img'

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
    #key = key[:-4]
    value = img.get_attribute('src')
    src[key] = value

for i, key in enumerate(src):
    urllib.request.urlretrieve(str(src[key]),"F:/Users/Henry/Coding/League of Icons/Icons/Other/Others_{}.jpg".format(key))
    c = Image.open('F:/Users/Henry/Coding/League of Icons/Icons/Other/Others_'+key+'.jpg')
    d = c.resize((120,120), resample=Image.BICUBIC)
    rgb_im = d.convert('RGB')
    rgb_im =  rgb_im.save('F:/Users/Henry/Coding/League of Icons/Icons/Other/Others_'+key+'.jpg')
from selenium import webdriver
from time import sleep
import json
from bs4 import BeautifulSoup

data = []
browser = webdriver.Chrome(executable_path='chromedriver.exe')

def exportjson(data):
    with open('crawlfile.json', 'w') as f:
        json.dump(data, f, indent=4)
        print('Data extracted')

def openpage(link):
    browser.get(link)
    return browser.page_source

def getamazondata(product, cate):
    name = product.h2.text.strip()
    url = 'http://amazon.com' + product.h2.a.get('href')
    try:
        price = product.find('span', class_= 'a-offscreen').text
    except:
        price = 'None'
    img = product.find('div', class_='a-section aok-relative s-image-square-aspect').img['src']
        
    record = {
            'Name': name,
            'URL': url,
            'Price': price,
            'ImageLink': img,
            'Category': cate
        }
    data.append(record)

def getdarveysdata(product,cate):
    name = product.find('div', class_='product-details').p.a.get('title')
    url = product.find('div', class_='product-details').p.a.get('href')
    try:
        price = product.find('div', class_= 'special-price').text
    except:
        price = 'None'
    img = product.a.img['src']
    if img == 'https://media.darveys.com/resized/?lv=51':
        img = 'https://i.pinimg.com/originals/f5/05/24/f50524ee5f161f437400aaf215c9e12f.jpg'    
    record = {
            'Name': name,
            'URL': url,
            'Price': price,
            'ImageLink': img,
            'Category': cate
        }
    data.append(record)

def main():
    clothes_categories = ['shirt', 'outwear', 'shorts', 'skirt', 'dress', 'jacket', 'pants']
    for i in range (len(clothes_categories)):
        for j in range (1,8):
            linkamazon = 'https://www.amazon.com/s?k=' + clothes_categories[i] + f'&page={j}'
            html_text = openpage(linkamazon)
            soup = BeautifulSoup(html_text, 'html.parser')
            products = soup.find_all('div', {'data-asin': True, 'data-component-type': 's-search-result'})
            cate = clothes_categories[i]
            if cate == 'jacket':
                cate = 'outwear'
            elif cate == 'pants':
                cate = 'short'
            elif cate == 'shorts':
                cate = 'short'
            for product in products:
                getamazondata(product, cate)
            linkdarveys = 'https://www.darveys.com/catalogsearch/result/?cat=2&q='+ clothes_categories[i] + f'&p={j}'
            html_text = openpage(linkdarveys)
            soup = BeautifulSoup(html_text, 'html.parser')
            products = soup.find_all('div', {'data-id': True, 'data-producttype': 'configurable'})
            for product in products:
                getdarveysdata(product, cate)
    exportjson(data)


if __name__ == '__main__':
    main()

sleep(5)
browser.close()


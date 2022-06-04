from unicodedata import category
import urllib.request
import json

#from the imageurl in crawl json file, download and store them to the database

with open('crawlfile.json', 'r') as f:
    temp = json.loads(f.read())
    for i in range(len(temp)):
        category = str(temp[i]['Category'])
        url = str(temp[i]['ImageLink'])
        full_path = 'crawldata/' + category + '/' + str(i).zfill(6) + '.jpg'
        urllib.request.urlretrieve(url, full_path)
        print(temp[i]['ImageLink'])
    
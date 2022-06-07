import json
import cv2

TRAINNO = 162331
VALIDNO = 52199
TRAINFOLDER = 'train/image/'
VALIDFOLDER = 'validation/image/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'
TRAINSAVEFOLDER = 'traindata/'
VALIDSAVEFOLDER = 'validdata/'
YOLOTRAINFOLDER = 'yolotraindataset/'
CRAWLFILE = 'crawlfile.json'
FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

#Prepare the data for training the yolov5 model
#Take images from DeepLearning dataset so that each categories has 5000 components

data = []
with open(CRAWLFILE, 'r') as f:
    temp = json.loads(f.read())
    for s in range(len(temp)):
        name = temp[s]['Name']
        cate = temp[s]['Category']
        url = temp[s]['URL']
        price = temp[s]['Price']
        record = {
            'Name': name,
            'URL': url,
            'Price': price,
            'Category': cate
        }
        data.append(record)
        PATH = 'crawldata/' + record['Category'] + '/' + str(s).zfill(6) + '.json'
        with open(PATH, 'w') as f:
            json.dump(record, f)
    print('Done')
    
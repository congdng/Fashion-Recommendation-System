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
YOLOVALIDFOLDER = 'yolovaliddataset/'
FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

#Prepare the data for training the yolov5 model
#Take images from DeepLearning dataset so that each categories has 1500 components

data = []
dataset = []
with open(VALIDJSONPATH, 'r') as f:
    temp = json.loads(f.read())
    for i in temp:
        if i == 'item':
            continue
        else:
            for s in range(VALIDNO):
                image = temp[i][s]['image']
                category = temp[i][s]['categories']
                record = {
                'Image': image,
                'Category': category,
                }
                data.append(record)

clothes_categories = ['shirt', 'outwear', 'short', 'skirt', 'dress']
for i in range (0,5):
    num = 0
    for j in range(VALIDNO):
        if (num < 1500):
            if data[j]['Category'] == clothes_categories[i]:
                num += 1
                if data[j]['Image'] not in dataset:
                    dataset.append(data[j]['Image'])
                    

print(len(dataset))
for k in range(len(dataset)):
    jsondataset = {
        "item": {},
        "info": []
    }
    filename = VALIDFOLDER + str(dataset[k])
    jsonname = 'validation/annos/' + str(dataset[k][0:6]) + '.json'
    img = cv2.imread(filename)
    cv2.imwrite(YOLOVALIDFOLDER + str(dataset[k]), img)
    imgheight, imgwidth, imgchannels = img.shape
    with open(jsonname, 'r') as f:
        temp = json.loads(f.read())
        for i in temp:
            if i == 'source' or i=='pair_id':
                continue
            else:
                box = temp[i]['bounding_box']
                bbox=[box[0],box[1],box[2],box[3]]
                xcenter = ((box[0] + box[2]) / 2) / imgwidth
                ycenter = ((box[1] + box[3]) / 2) / imgheight
                width = (box[2] - box[0]) / imgwidth
                height = (box[3] - box[1]) / imgheight
                cat = temp[i]['category_id']
                if (cat == 1) | (cat == 2) | (cat == 5) | (cat == 6):
                    cat = 0
                elif (cat == 3) | (cat == 4):
                    cat = 1
                elif (cat == 7) | (cat == 8):
                    cat = 2
                elif (cat == 9):
                    cat = 3
                else:
                    cat = 4
                jsondataset['info'].append({
                    "categories": cat,
                    "xcenter": xcenter,
                    "ycenter": ycenter,
                    "width": width,
                    "height": height
                })
                with open(YOLOVALIDFOLDER + str(dataset[k][0:6]) + '.txt', 'w+') as f:
                    for i in range(len(jsondataset['info'])):
                        f.write(str(jsondataset['info'][i]['categories']) + ' ' + 
                                str(jsondataset['info'][i]['xcenter']) + ' ' +
                                str(jsondataset['info'][i]['ycenter']) + ' ' +
                                str(jsondataset['info'][i]['width']) + ' ' +
                                str(jsondataset['info'][i]['height']) +
                                '\n')
                    f.close()
    jsondataset.clear()

print('DONE')

#python train.py --batch 64 --epochs 10 --data data.yaml --cfg yolov5s.yaml --name Model --weight /runs/train/Model5/weights/best.pt
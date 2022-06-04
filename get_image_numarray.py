import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import gradient_descent_v2
import json
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.optimizers import adam_v2

TRAINNO = 162331
VALIDNO = 52199
TRAINFOLDER = 'train/image/'
VALIDFOLDER = 'validation/image/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'
TRAINSAVEFOLDER = 'traindata/'
VALIDSAVEFOLDER = 'validdata/'
FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

#With the already trained model, run and get the 2048-number array from the crawl images.

dataset = {
    "item": {},
    "info": []
}

def get_embedding(model, imagename):
    img = image.load_img(imagename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    return model.predict(x).reshape(-1)

restored_model = tf.keras.models.load_model('bestmodel3.h5')
restored_model.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])
secondmodel = Model(inputs = restored_model.input,
                                 outputs= restored_model.layers[266].output)
secondmodel.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])

with open('crawlfile.json', 'r') as f:
    temp = json.loads(f.read())
    for i in range(len(temp)):
        category = str(temp[i]['Category'])
        name = str(temp[i]['Name'])
        url = str(temp[i]['URL'])
        price = str(temp[i]['Price'])
        imagelink = str(temp[i]['ImageLink'])
        img = 'crawldata/' + category + '/' + str(i).zfill(6) + '.jpg'
        notedarray = get_embedding(secondmodel, img).tolist()
        dataset['info'].append({
                'no': i,
                'categories': category,
                'name': name,
                'url': url,
                'imagelink': img,
                'notedarray': notedarray
                })

#print(dataset)
with open('notedarray.json', 'w') as f:
    json.dump(dataset, f)


    

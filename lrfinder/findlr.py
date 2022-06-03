import numpy as np
from lrfinder import LRFinder
import tensorflow as tf
import numpy as np
import pandas as pd
import json 
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.optimizers import adam_v2
from keras import callbacks
from keras.applications.mobilenet_v3 import MobileNetV3Large
from keras.applications.resnet import ResNet50
K = tf.keras.backend

TRAINNO = 81439
VALIDNO = 14349
TRAINFOLDER = 'train/image/'
VALIDFOLDER = 'validation/image/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'
TRAINSAVEFOLDER = 'C:/Users/ADMIN/Desktop/Project/traindata'
VALIDSAVEFOLDER = 'C:/Users/ADMIN/Desktop/Project/validdata'
FAST_RUN = False
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

def outputmodel(model):
    outmodel = model.output
    #outmodel = Dropout(0.3)(outmodel)
    #outmodel = Flatten(name = 'flatten')(outmodel)
    outmodel = GlobalAveragePooling2D()(outmodel)
    outmodel = BatchNormalization()(outmodel)
    outmodel = Dense(2048)(outmodel)
    outmodel = Activation('relu')(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(5)(outmodel)
    outmodel = Activation('softmax')(outmodel)    
    return outmodel

model = MobileNetV3Large(weights = 'imagenet', 
    include_top = False,       
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

for layer in model.layers:
    layer.trainable = False

OutputModel = outputmodel(model)
Mobinetmodel = Model(inputs = model.input, outputs = OutputModel)

for (i,layer) in enumerate(Mobinetmodel.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
print(Mobinetmodel.summary())

traingene = ImageDataGenerator(
    rotation_range=20,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
validgene = ImageDataGenerator(rescale=1./255)

train_set = traingene.flow_from_directory(
        TRAINSAVEFOLDER,
        target_size = IMAGE_SIZE,
        batch_size = 64,
        class_mode='categorical')
test_set = validgene.flow_from_directory(
        VALIDSAVEFOLDER,
        target_size = IMAGE_SIZE,
        batch_size = 32,
        class_mode='categorical',
        shuffle=False)

Mobinetmodel.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 1e-4) , metrics = ['accuracy'])
lr_finder = LRFinder(Mobinetmodel)
lr_finder.find(train_set, start_lr=1e-6, end_lr=1, epochs=5,steps_per_epoch=371)
best_lr = lr_finder.get_best_lr(sma=20)

print(best_lr)


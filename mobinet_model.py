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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

def outputmodel(model):
    outmodel = model.output
    #outmodel = Dropout(0.3)(outmodel)
    outmodel = Flatten(name = 'flatten')(outmodel)
    #outmodel = GlobalAveragePooling2D()(outmodel)
    outmodel = BatchNormalization()(outmodel)
    outmodel = Dense(2048)(outmodel)
    outmodel = Activation('relu')(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(1024)(outmodel)
    outmodel = Activation('relu')(outmodel)
    outmodel = Dropout(0.25)(outmodel)
    outmodel = Dense(512)(outmodel)
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

mc = callbacks.ModelCheckpoint('bestmodel.h5', monitor = 'accuracy', mode = 'max', save_best_only = True)
es = callbacks.EarlyStopping(monitor = 'accuracy', verbose = 1,
                                        mode = "max", patience = 10,
                                        restore_best_weights = True)
#sgd = gradient_descent_v2.SGD(learning_rate = 1e-4, momentum = 0.9, nesterov = True)
Mobinetmodel.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])

epochs = 10 if FAST_RUN else 200
history = Mobinetmodel.fit(
    train_set, 
    epochs = epochs,
    validation_data = test_set,
    verbose = 1,
    callbacks=[mc,es]
)

Mobinetmodel.save('model.h5')
accuracy = Mobinetmodel.history.history['accuracy']
val_accuracy = Mobinetmodel.history.history['val_accuracy']

loss = Mobinetmodel.history.history['loss']
val_loss = Mobinetmodel.history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()




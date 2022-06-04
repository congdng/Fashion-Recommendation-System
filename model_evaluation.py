import numpy as np
import tensorflow as tf
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

#Check the accuracy and continue training phase if the accuracy is not as demand

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

mc = callbacks.ModelCheckpoint('bestmodel1.h5', monitor = 'accuracy', mode = 'max', save_best_only = True)
es = callbacks.EarlyStopping(monitor = 'accuracy', verbose = 1,
                                        mode = "max", patience = 10,
                                        restore_best_weights = True)

restored_model = tf.keras.models.load_model('model.h5')
#print(restored_model.summary())
restored_model.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])

epochs = 3 if FAST_RUN else 200
history = restored_model.fit(
    train_set, 
    epochs=epochs,
    validation_data = test_set,
    verbose=1,
    callbacks=[mc,es]
)

restored_model.save('model1.h5')
accuracy = restored_model.history.history['accuracy']
val_accuracy = restored_model.history.history['val_accuracy']

loss = restored_model.history.history['loss']
val_loss = restored_model.history.history['val_loss']

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



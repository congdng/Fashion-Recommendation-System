import tensorflow as tf
import json
import pandas as pd
from keras.preprocessing import image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from keras.models import Model
from keras.optimizers import gradient_descent_v2
from keras.optimizers import adam_v2

TRAINNO = 162331
VALIDNO = 52199
TRAINFOLDER = 'train/image/'
VALIDFOLDER = 'validation/image/'
TRAINJSONPATH = 'sum.json'
VALIDJSONPATH = 'validsum.json'
TRAINSAVEFOLDER = 'traindata/'
VALIDSAVEFOLDER = 'validdata/'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

#Run the similarity calculation algorithm (K-NN)
 
def get_embedding(model, imagename):
    img = image.load_img(imagename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    return model.predict(x).reshape(-1)

with open('notedarray.json', 'r') as f:
    temp = json.loads(f.read())
    for i in temp:
        if i == 'item':
            continue
        else:
            imagelink = []
            categories = []
            notedarray = []
            for s in range(len(temp[i])):
                imagelink.append(str(temp[i][s]['imagelink']))
                categories.append(str(temp[i][s]['categories']))
                fixarray = str(temp[i][s]['notedarray']).replace('[', '')
                fixarray = fixarray.replace(']', '')
                notedarray.append(fixarray.split(','))
                df = pd.DataFrame({
                    'imagelink': imagelink,
                    'category': categories,
                    'notedarray': notedarray
                })
    print(df)

restored_model = tf.keras.models.load_model('bestmodel.h5')
restored_model.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])
secondmodel = Model(inputs = restored_model.input,
                                 outputs= restored_model.layers[266].output)
secondmodel.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])
clothes_categories = ['shirt', 'outwear', 'short', 'skirt', 'dress']

for i in range(len(clothes_categories)):
    select = df.loc[df['category'] == clothes_categories[i - 1]]
    map_embeddings = select['notedarray']
    print(map_embeddings)
    df_embs = map_embeddings.apply(pd.Series)
    print(df_embs.shape)
    neighbors = NearestNeighbors(n_neighbors = 5, algorithm='brute', metric='euclidean')
    neighbors.fit(df_embs)
    select = select.iloc[0:0]
    #print(select)
    knnPickle = open('crawldata/' + clothes_categories[i - 1] + '/knn.pkl', 'wb')
    pickle.dump(neighbors, knnPickle)
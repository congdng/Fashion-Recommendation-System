import numpy as np
import torch
import cv2
import os
import pickle
import numpy as np
import json
from keras.preprocessing import image
from keras.applications.mobilenet_v3 import preprocess_input
import tensorflow as tf
from keras.models import Model
import streamlit as st
from keras.optimizers import adam_v2
from PIL import Image

TRAINNO = 32400
VALIDNO = 31354
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

#Run by streamlit run main.py
record =[]
def get_embedding(model, imagename):
    img = image.load_img(imagename, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    x   = image.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

def save_file(uploaded_file):
    try:
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def readfile(PATH):
    with open(PATH, 'r') as f:
        temp = json.loads(f.read())
        name = temp['Name']
        cate = temp['Category']
        url = temp['URL']
        price = temp['Price']
        record = {
            'Name': name,
            'URL': url,
            'Price': price,
            'Category': cate
        }
    return record

restored_model = tf.keras.models.load_model('bestmodel3.h5')
secondmodel = Model(inputs = restored_model.input,
                                 outputs= restored_model.layers[266].output)
#print(secondmodel.summary())
restored_model.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])
secondmodel.compile(loss = 'categorical_crossentropy', optimizer = adam_v2.Adam(learning_rate = 2.5e-4) , metrics = ['accuracy'])
yolomodel = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/Model11/weights/best.pt')

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        # display image
        uploadimage = cv2.imread(uploaded_file.name)
        st.image(uploaded_file)
        #imagearray = get_embedding(secondmodel, uploaded_file.name)
        filename = uploaded_file.name[0:6]
        #image = cv2.imread('validation/image/005879.jpg')
        detections = yolomodel(uploadimage)
        results = detections.pandas().xyxy[0].to_dict(orient = 'records')
        x = np.array(results)
        choicearray = ()
        if len(x) == 0:
            st.write('Not found')
        else:
            st.write('Found ' + str(len(x)) + ' results in your picture')
            cols = st.columns(len(x))
            for i in range (len(x)):
                name = results[i]['name']
                clas = results[i]['class']
                xmin = int(results[i]['xmin'])
                ymin = int(results[i]['ymin'])
                xmax = int(results[i]['xmax'])
                ymax = int(results[i]['ymax'])
                #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0 ,0), 2)
                #cv2.putText(image, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 255, 255), 1 )
                crop_image = uploadimage[ymin:ymax, xmin:xmax]
                cv2.imwrite('detect/' + filename + str(i) + '.jpg', crop_image)
            for i, col in enumerate(cols):
                col.image('detect/' + filename + str(i) + '.jpg')
                col.write(str(i + 1))
                choicearray += (i + 1,)
            option = st.selectbox('Choose with clothes you want to find?', choicearray)
            st.write('You selected:', option)
            #df['Power']=df['Power'].apply(lambda row: float(row))
            loaded_model = pickle.load(open('crawldata/' + results[option - 1]['name'] + '/knn.pkl', 'rb'))
            imagearray = get_embedding(secondmodel, 'detect/' + filename + str(option - 1) + '.jpg')
            thislist = sorted(filter(lambda x: os.path.isfile(os.path.join('crawldata/'+ results[option - 1]['name'] + '/', x)), os.listdir('crawldata/'+ results[option - 1]['name'] + '/')))
            distance, indices = loaded_model.kneighbors([imagearray])
            print(indices)
            for j in range(4):
                print('crawldata/' + results[option - 1]['name'] + '/' + str(thislist[indices[0][j]*2]))
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image('crawldata/' + results[option - 1]['name'] + '/' + thislist[indices[0][0]*2])
                PATH = 'crawldata/'+ results[option - 1]['name'] + '/' + thislist[indices[0][0]*2 + 1]
                record = readfile(PATH)
                st.write(record['Name'])
                st.write(record['Price'])
                link = record['URL']
                link = '[Click here](' + link +  ')'
                st.markdown(link, unsafe_allow_html=True)
            with col2:
                st.image('crawldata/' + results[option - 1]['name'] + '/' + thislist[indices[0][1]*2])
                PATH = 'crawldata/'+ results[option - 1]['name'] + '/' + thislist[indices[0][1]*2 + 1]
                record = readfile(PATH)
                st.write(record['Name'])
                st.write(record['Price'])
                link = record['URL']
                link = '[Click here](' + link +  ')'
                st.markdown(link, unsafe_allow_html=True)
            with col3:
                st.image('crawldata/' + results[option - 1]['name'] + '/' + thislist[indices[0][2]*2])
                PATH = 'crawldata/'+ results[option - 1]['name'] + '/' + thislist[indices[0][2]*2 + 1]
                record = readfile(PATH)
                st.write(record['Name'])
                st.write(record['Price'])
                link = record['URL']
                link = '[Click here](' + link +  ')'
                st.markdown(link, unsafe_allow_html=True)
            with col4:
                st.image('crawldata/' + results[option - 1]['name'] + '/' + thislist[indices[0][3]*2])
                PATH = 'crawldata/'+ results[option - 1]['name'] + '/' + thislist[indices[0][3]*2 + 1]
                record = readfile(PATH)
                st.write(record['Name'])
                st.write(record['Price'])
                link = record['URL']
                link = '[Click here](' + link +  ')'
                st.markdown(link, unsafe_allow_html=True)



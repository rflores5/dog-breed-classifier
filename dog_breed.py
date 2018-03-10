# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:32:59 2018

@author: Owner
"""
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from extract_bottleneck_features import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50  
from keras.applications.resnet50 import preprocess_input, decode_predictions                
from tqdm import tqdm
import cv2  

# load the list of dog breeds to predict from
file2 = open("breeds.txt","r")
dog_names = file2.read().split()

# load features from InceptionsV3Data
features = np.load("bottleneck_features/DogInceptionV3Data.npz")
train_inception = features["train"]
valid_inception = features["valid"]
test_inception = features["test"]

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


# build model 
shape=train_inception.shape[1:]                    #the shape of the input is the output of the inception model
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=shape)) #global average pooling layers redue the number of parameters and speed up training
model.add(Dense(2048,activation='sigmoid')) # Adding 3 fully connected layers
model.add(Dropout(.25))                          # Adding Droupout layers to reduce overfitting                   
model.add(Dense(256,activation='sigmoid'))       # during a previouis CNN exercise I found sigmoid performs better than relu in the connected layers
model.add(Dropout(.3))
model.add(Dense(133,activation='softmax'))    # last layer has output of 133 nodes for the 133 options for dog breeds, 
                                            # softmax converts output to probabilities
model.load_weights("saved_models/weights.best.inception.hdf5")

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def face_detector(img_path):
    img4 = cv2.imread(img_path)
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img2 = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img2))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def predict_dog_breed(img_path):
    features = extract_InceptionV3(path_to_tensor(img_path))
    predictions = model.predict(features)
    return dog_names[np.argmax(predictions)]

def human_dog_breed(img_path):
    breed = predict_dog_breed(img_path)
    breed = breed.split(".")[1]
    plt.imshow(mpimg.imread(img_path))
    plt.show()
    if dog_detector(img_path)==True:
        print("This dog looks like a %s" % breed)
    elif face_detector(img_path)==True:
        print("This human looks like a %s" % breed )
    else:
        print("This is neither a dog nor human, but looks like a %s" % breed)
        
test_images = glob('test\*')
for img3 in test_images:
    human_dog_breed(img3)
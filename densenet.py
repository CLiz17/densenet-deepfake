import os
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns #graphs
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential #model layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, AvgPool2D, GlobalAveragePooling2D, MaxPool2D, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, ReLU, concatenate

def load_images_and_labels(folder_path):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        
        train_split = 0.8
        
        image_files = os.listdir(category_path)
        num_train_samples = int(len(image_files) * train_split)
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(category_path, image_file)
            
            image = cv2.imread(image_path)
            
            if i < num_train_samples:
                train_images.append(image)
                train_labels.append(category)
            else:
                test_images.append(image)
                test_labels.append(category)
           
    return (
        np.array(train_images), np.array(train_labels),
        np.array(test_images), np.array(test_labels)
    )

folder_path = "/home/liz/densenet-deepfake/dataset"
x_train, y_train, x_test, y_test = load_images_and_labels(folder_path)


fpath = "/home/liz/densenet-deepfake/dataset"
random_seed = 42
categories = os.listdir(fpath)

def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        for image_name in os.listdir(fpath+"/"+category):
            img = cv2.imread(fpath+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_array = Image.fromarray(img, 'RGB')
            resized_img = img_array.resize((150, 150))

            img_lst.append(np.array(resized_img))
            labels.append(index)
    return img_lst, labels

images, labels = load_images_and_labels(categories)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
print(type(images),type(labels))
images = np.array(images)
labels = np.array(labels)

print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
print(type(images),type(labels))
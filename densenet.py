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

import os
import cv2
import numpy as np
from PIL import Image

def load_images_and_labels(folder_path):
    img_lst = []
    labels = []
    categories = os.listdir(folder_path)

    for index, category in enumerate(categories):
        category_path = os.path.join(folder_path, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = Image.fromarray(img, 'RGB')
            resized_img = img_array.resize((150, 150))
            img_lst.append(np.array(resized_img))
            labels.append(index)

    images = np.array(img_lst)
    labels = np.array(labels)

    return images, labels

folder_path = "/home/liz/densenet-deepfake/dataset"
images, labels = load_images_and_labels(folder_path)

print("No. of images loaded:", len(images))
print("No. of labels loaded:", len(labels))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Data types - Images:", type(images), ", Labels:", type(labels))
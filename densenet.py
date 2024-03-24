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

def load_images_and_labels(fpath):
    img_lst = []
    labels = []
    categories = os.listdir(fpath)

    for index, category in enumerate(categories):
        category_path = os.path.join(fpath, category)
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

fpath = "/home/liz/densenet-deepfake/dataset"
images, labels = load_images_and_labels(fpath)

print("No. of images loaded:", len(images))
print("No. of labels loaded:", len(labels))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Data types - Images:", type(images), ", Labels:", type(labels))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)    

train_generator = train_datagen.flow_from_directory(
    fpath,
    target_size = (150,150),
    batch_size = 32,subset = 'training',
    class_mode = 'binary'
)
     
validation_generator = train_datagen.flow_from_directory(
    fpath,
    target_size = (150,150),
    batch_size = 32,subset = 'validation',
    class_mode = 'binary',shuffle=False
)

dense_model = DenseNet121(input_shape=(150,150,3),include_top=False,weights="imagenet")
for layer in dense_model.layers:
    layer.trainable=False
model=Sequential()
model.add(dense_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(2048,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.summary()

import tensorflow as tf
OPT    = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=OPT)

hist = model.fit(
    train_generator,
    epochs = 10,
    validation_data = validation_generator
)

model.save('model_weights.h5')

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Densenet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left', bbox_to_anchor=(1,1))
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Densenet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left', bbox_to_anchor=(1,1))
plt.show()

y_true = validation_generator.classes
y_pred = (model.predict(validation_generator) > 0.5).astype("int32")
cm=confusion_matrix(y_true, y_pred)

sns.heatmap(cm,cmap="plasma",fmt="d",annot=True)
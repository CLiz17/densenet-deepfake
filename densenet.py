import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.applications.densenet import DenseNet121

def load_images_and_labels(fpath):
    images = []
    labels = []

    image_files = os.listdir(fpath)
    for image_name in image_files:
        image_path = os.path.join(fpath, image_name)
        img = cv2.imread(image_path)
        images.append(img)
        label = extract_label_from_filename(image_name)
        labels.append(label)

    return np.array(images), np.array(labels)

def extract_label_from_filename(filename):
    if "_0" in filename.lower():
        return 0
    elif "_1" in filename.lower():
        return 1

fpath = "/content/drive/MyDrive/dataset"
images, labels = load_images_and_labels(fpath)


print("No. of images loaded:", len(images))
print("No. of labels loaded:", len(labels))
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Training set size:", len(x_train)+len(y_train))
print("Validation set size:", len(x_test)+len(y_test))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=32
)

validation_generator = train_datagen.flow(
    x_test,
    y_test,
    batch_size=32
)

dense_model = DenseNet121(input_shape=(150,150,3), include_top=False, weights='imagenet')
for layer in dense_model.layers:
    layer.trainable=True

model = Sequential([
    dense_model,
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(2048, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1024, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=(x_test, y_test)
)

model.save('model3.h5')

plt.plot(hist.history['accuracy'], label='Train Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Densenet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Densenet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

y_pred = model.predict(x_test) > 0.5
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap="plasma", fmt="d", annot=True)

model = load_model("/content/model3.h5")

def predict(file_path):
    img = load_img(file_path, target_size=(150, 150))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction < 0.5:
        return "The image is a deepfake"
    else:
        return "The image is of a real person"
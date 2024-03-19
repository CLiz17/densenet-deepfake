import os
import cv2
import numpy as np

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
    print(train_images)
           
    return (
        np.array(train_images), np.array(train_labels),
        np.array(test_images), np.array(test_labels)
    )

folder_path = "/home/liz/densenet-deepfake/dataset"
x_train, y_train, x_test, y_test = load_images_and_labels(folder_path)


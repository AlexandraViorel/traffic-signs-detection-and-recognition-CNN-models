import os
import random
from skimage import io, transform
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image

def load_data(dataset_path):
    image_list = []
    labels_list = []
    classes = 75

    try:
        for traffic_sign_class in range(classes):
            print(f"[INFO] Processing {traffic_sign_class} ...")
            path = os.path.join(dataset_path, str(traffic_sign_class))
            traffic_sign_class_images = os.listdir(path)
            for image in traffic_sign_class_images:
                try:
                    img = Image.open(path + "/" + image)
                    img = img.resize((45, 45))
                    img = np.array(img)
                    image_list.append(img)
                    labels_list.append(traffic_sign_class)
                except Exception as err:
                    print(f"[ERROR]: On loading image!")

    except Exception as err:
        print(f"[ERROR]: {err}")

    image_list = np.array(image_list, dtype=np.float16) / 255.0 
    labels_list = np.array(labels_list)
    return image_list, labels_list

def split_train_val():
    print("[INFO] loading train and val images...")
    dataset_path = r"D:\Faculty materials\bachelors\datasets\GermanAndBelgianTS"

    image_list, labels_list = load_data(dataset_path)
    x_train, x_val, y_train, y_val = train_test_split(image_list, labels_list, test_size=0.2, random_state=42, shuffle=True)


    y_train = to_categorical(y_train, 75)
    y_val = to_categorical(y_val, 75)

    return x_train, x_val, y_train, y_val

def load_test_dataset():
    csv_path_test = r"D:\Faculty materials\bachelors\datasets\GTSRB\Test.csv"
    dataset_path = r"D:\Faculty materials\bachelors\datasets\GTSRB"

    x_test, y_test = load_data(dataset_path, csv_path_test)
    y_test = to_categorical(y_test, 75)

    return x_test, y_test
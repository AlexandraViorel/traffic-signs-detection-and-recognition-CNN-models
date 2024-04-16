import os
import random
from skimage import io, transform
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(dataset_path, csv_path):
    image_list = []
    labels_list = []
    classes = 43

    rows = open(csv_path).read().strip().split("\n")[1:]

    for (i, row) in enumerate(rows):
        if i > 0 and i % 1000 == 0:
            print(f"[INFO] processed {i} images...")
        (label, image_path) = row.strip().split(",")[-2:]
        image_path = os.path.join(dataset_path, image_path)
        image = io.imread(image_path)
        image = transform.resize(image, (45, 45))
        image_list.append(image)
        labels_list.append(label)

    image_list = np.array(image_list, dtype=np.float32) / 255.0
    labels_list = np.array(labels_list)
    return image_list, labels_list

def split_train_val():
    print("[INFO] loading train and val images...")
    csv_path_train = r"D:\Faculty materials\bachelors\datasets\GTSRB\Train.csv"
    dataset_path = r"D:\Faculty materials\bachelors\datasets\GTSRB"

    image_list, labels_list = load_data(dataset_path, csv_path_train)
    x_train, x_val, y_train, y_val = train_test_split(image_list, labels_list, test_size=0.2, random_state=27)


    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)

    return x_train, x_val, y_train, y_val

def load_test_dataset():
    csv_path_test = r"D:\Faculty materials\bachelors\datasets\GTSRB\Test.csv"
    dataset_path = r"D:\Faculty materials\bachelors\datasets\GTSRB"

    x_test, y_test = load_data(dataset_path, csv_path_test)
    y_test = to_categorical(y_test, 43)

    return x_test, y_test

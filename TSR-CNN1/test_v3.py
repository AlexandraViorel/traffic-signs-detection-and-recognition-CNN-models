import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import load_model
import os
from skimage import io, transform
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
    labels_list = np.array(labels_list, dtype=np.int32)  
    return image_list, labels_list

def load_test_dataset():
    csv_path_test = r"D:\Faculty materials\bachelors\datasets\GTSRB\Test.csv"
    dataset_path = r"D:\Faculty materials\bachelors\datasets\GTSRB"

    x_test, y_test = load_data(dataset_path, csv_path_test)
    y_test = to_categorical(y_test, 43)

    return x_test, y_test

def evaluate_models(models_directory, x_test, y_test, batch_size):
    num_classes = len(np.unique(np.argmax(y_test, axis=1)))
    
    model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]
    
    for model_file in model_files:
        print(f"[INFO] Evaluating model {model_file}...")
        model_path = os.path.join(models_directory, model_file)
        model = load_model(model_path)
        
        y_pred = model.predict(x_test, batch_size=batch_size)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        print(f"Sample predictions: {y_pred_classes[:10]}")
        print(f"Sample actual values: {y_test_classes[:10]}")
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        print(f'Accuracy: {accuracy}')
        
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        cm_df = pd.DataFrame(cm_normalized, index=range(num_classes), columns=range(num_classes))
        
        non_zero_cells = cm_df[cm_df != 0].stack().reset_index()
        non_zero_cells.columns = ['Actual', 'Predicted', 'Value']
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_df, annot=False, cmap='Blues', cbar=True)
        plt.xticks(ticks=np.arange(0, num_classes, 5), labels=np.arange(0, num_classes, 5), rotation=90, fontsize=12)
        plt.yticks(ticks=np.arange(0, num_classes, 5), labels=np.arange(0, num_classes, 5), fontsize=12)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('Actual', fontsize=16)
        plt.title('Normalized Confusion Matrix', fontsize=20)
        plt.show()
        
        report = classification_report(y_test_classes, y_pred_classes)
        print(report)

x_test, y_test = load_test_dataset()

models_directory = "TSRNet1\models"
batch_size = 32
evaluate_models(models_directory, x_test, y_test, batch_size)

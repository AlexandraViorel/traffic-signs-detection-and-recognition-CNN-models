import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import to_categorical
import os 

df = pd.read_csv(r'D:\Faculty materials\bachelors\datasets\GermanAndBelgianTS_Test\Test_cleaned.csv')

base_dir = r'D:\Faculty materials\bachelors\datasets\GermanAndBelgianTS_Test\\'
df['Path'] = base_dir + df['Path'].astype(str)

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0  
    return img

target_size = (45, 45)  
X_test = np.array([preprocess_image(row['Path'], target_size) for index, row in df.iterrows()])
y_test = df['ClassId'].values

num_classes = len(np.unique(y_test))
y_test = to_categorical(y_test, num_classes)

models_directory = "TSRNet3New\models"

model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]
# model_files = [r'D:\Faculty materials\BACHELORS-THESIS\TSR-CNN3\TSRNet3Original\models\fragrant-sweep-8.h5']

for model_file in model_files:
    print(f"[INFO] Evaluating model {model_file}...")

    model_path = os.path.join(models_directory, model_file)
    model = load_model(model_path)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test_classes, y_pred_classes)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_df = pd.DataFrame(cm_normalized, index=range(num_classes), columns=range(num_classes))

    non_zero_cells = cm_df[cm_df != 0].stack().reset_index()
    non_zero_cells.columns = ['Actual', 'Predicted', 'Value']

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_df, annot=False, cmap='Blues', cbar=True, xticklabels=range(num_classes), yticklabels=range(num_classes))

    plt.xticks(ticks=np.arange(0, num_classes, 5), labels=np.arange(0, num_classes, 5), rotation=90, fontsize=12)
    plt.yticks(ticks=np.arange(0, num_classes, 5), labels=np.arange(0, num_classes, 5), fontsize=12)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    report = classification_report(y_test_classes, y_pred_classes)
    print(report)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Accuracy: {accuracy}')
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model
from data_processing import load_test_dataset
import os
import numpy as np

BS=32
sign_labels = open("D:\Faculty materials\BACHELORS-THESIS\SignsNames.csv").read().strip().split("\n")[1:]
sign_labels = [l.split(",")[1] for l in sign_labels]

x_test, y_test = load_test_dataset()

models_directory = "TSRNet1\models"

model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]
print(len(model_files))
# model_files = ["earnest-sweep-2.h5", "mild-sweep-11.h5", "ancient-sweep-8.h5", "copper-sweep-7.h5", "ancient-sweep-6.h5", "rosy-sweep-4.h5", "brisk-sweep-3.h5", "pleasant-sweep-1.h5"]

for model_file in model_files:
    print(f"[INFO] Evaluating model {model_file}...")

    model_path = os.path.join(models_directory, model_file)
    model = load_model(model_path)
    predictions = model.predict(x_test, batch_size=BS)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    # print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=sign_labels))
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f'Accuracy: {accuracy}')
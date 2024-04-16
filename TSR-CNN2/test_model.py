from sklearn.metrics import classification_report
from keras.models import load_model
from data_processing import load_test_dataset
import os

BS=32
sign_labels = open("D:\Faculty materials\BACHELORS-THESIS\SignsNames.csv").read().strip().split("\n")[1:]
sign_labels = [l.split(",")[1] for l in sign_labels]

x_test, y_test = load_test_dataset()

models_directory = "TSRNet2\models"

model_files = [f for f in os.listdir(models_directory) if f.endswith('.h5')]

for model_file in model_files:
    print(f"[INFO] Evaluating model {model_file}...")

    model_path = os.path.join(models_directory, model_file)
    model = load_model(model_path)
    predictions = model.predict(x_test, batch_size=BS)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=sign_labels))
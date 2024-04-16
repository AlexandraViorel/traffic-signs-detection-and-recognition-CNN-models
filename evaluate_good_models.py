from keras.models import load_model
from PIL import Image
import numpy as np

sign_labels = open("SignsNames.csv").read().strip().split("\n")[1:]
sign_labels = [l.split(",")[1] for l in sign_labels]

good_models = ["TSR-CNN2\TSRNet2\models\silvery-sweep-1.h5", "TSR-CNN2\TSRNet2\models\glad-sweep-2.h5", "TSR-CNN1\TSRNet1\models\dandy-sweep-1.h5", "TSR-CNN1\TSRNet1\models\magic-sweep-2.h5",
               "TSR-CNN1\TSRNet1\models\dutiful-sweep-4.h5", r"TSR-CNN1\TSRNet1\models\fluent-sweep-1.h5", "TSR-CNN1\TSRNet1\models\expert-sweep-1.h5", r"TSR-CNN1\TSRNet1\models\avid-sweep-2.h5",
               "TSR-CNN1\TSRNet1\models\young-sweep-1.h5", "TSR-CNN1\TSRNet1\models\lilac-sweep-1.h5"]

images = ["general_caution.jpg", "stop.jpg"]
l = []

for i in images:
    img = Image.open(i)
    img = img.resize((45, 45))
    img = np.array(img)
    l.append(img)

image_list = np.array(l, dtype=np.float16) / 255.0

for model_path in good_models:
    print(f"[INFO] Testing model {model_path}")
    model = load_model(model_path)
    predictions = model.predict(image_list)

    for i, prediction in enumerate(predictions):
        # Get the index of the highest probability
        predicted_index = np.argmax(prediction)
        # Fetch the sign label using the predicted index
        predicted_label = sign_labels[predicted_index]
        print(f"Image: {images[i]}, Prediction: {predicted_label}")
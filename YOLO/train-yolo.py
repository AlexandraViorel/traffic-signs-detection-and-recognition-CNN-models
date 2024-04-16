from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data=r'D:\Faculty materials\bachelors\datasets\GTSDB\data.yaml', epochs=50, verbose=True, plots=True, optimizer='SGD')
metrics = model.val()

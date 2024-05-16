from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data=r'D:\Faculty materials\bachelors\datasets\GTSDB\data.yaml', epochs=200, verbose=True, plots=True, optimizer='SGD', save_dir=r'D:\Faculty materials\BACHELORS-THESIS\YOLO\runs\detect\train200')
metrics = model.val()

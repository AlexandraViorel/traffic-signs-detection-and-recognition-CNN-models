from ultralytics import YOLO

model = YOLO(r'D:\Faculty materials\BACHELORS-THESIS\runs\detect\train\weights\best.pt')

results = model([r'my-imgs\1.jpg', r'my-imgs\2.jpg', r'my-imgs\3.jpg', r'my-imgs\4.jpg', r'my-imgs\5.jpg', r'my-imgs\6.jpg', r'my-imgs\7.jpg'])  # return a list of Results objects
i=1
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename=f'result{i}.jpg')
    i+=1
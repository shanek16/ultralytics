from ultralytics import YOLO

model = YOLO("./runs/detect/train2/weights/best.pt")
source = '../../data/safety/video/forklift_accident1.mp4'
name = source.split('/')[-1][:-4]

results = model.predict(
    source=source, 
    # project='safety',
    # name=name,
    save=True,
    stream=True,
    imgsz=640, 
    conf=0.5
    )
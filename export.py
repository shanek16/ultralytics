from ultralytics import YOLO

# weights = '../runs/detect/train3/weights/best.pt' # Lbest.pt
weights = '../weights/detect/YOLOL.pt' # Lbest.pt

# Load a model
model = YOLO(weights)  # load a custom trained

# Export the model
model.export(format='engine', imgsz=640, simplify=True, half=False, device='0')
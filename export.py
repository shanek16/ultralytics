from ultralytics import YOLO

weights = '../runs/detect/train3/weights/best.pt' # Lbest.pt

# Load a model
model = YOLO(weights)  # load a custom trained

# Export the model
model.export(format='onnx', imgsz=640, batch=1, simplify=True, half=False)
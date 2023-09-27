import argparse
from ultralytics import YOLO
source = '../../data/safety/video/forklift_accident1.mp4'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--transfer",
    action='store_true',
    help="True--> transfer learning with pretrained model(COCO). \nFalse--> learn from scratch",
)
args = parser.parse_args()

if args.transfer:
    # Load pretrained YOLO model and transfer learn
    model = YOLO("yolov8s.pt")
    print('training with pretrained weight...')
else:
    # Create a new YOLO model from scratch
    model = YOLO('yolov8s.yaml')
    print('training from scratch...')

results = model.train(
    data="safety.yaml",
    epochs=300,
    imgsz=640,
    device='0,1', 
    batch=32
    )

results = model.val()

results = model(
    source=source,
    save=True,
    imgsz=640, 
    conf=0.5)
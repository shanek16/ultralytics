import cv2
from yolov8.utils.general import non_max_suppression
from yolov8.utils.torch_utils import select_device
from yolov8.models.experimental import attempt_load
from yolov8.utils.datasets import LoadStreams

# Load YOLOv8 model
# weights = './runs/detect/train2/weights/best.pt'
weights = '/home/swkim/Project/yolov8_tracking/runs/detect/train3/weights/best.engine'
device = select_device('')
model = attempt_load(weights, map_location=device)

# Load input video
# source = '../../data/safety/video/forklift_accident1.mp4'
source = '../../data/safety/video/ladder_short.mp4'
cap = cv2.VideoCapture(source)

# Define output video codec and fps
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video writer
output_file = 'predicted_video.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on frame
    img = model.preprocess(frame)
    pred = model(img)[0]
    pred = non_max_suppression(pred)

    # Draw bounding boxes on frame
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = model.xywh2xyxy(det[:, :4])
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                color = (255, 0, 0)
                line_thickness = 3
                xyxy = [int(x) for x in xyxy]
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=color,
                              thickness=line_thickness)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1e-3 * frame.shape[0], color=color,
                            thickness=2)

    # Write frame with object detection results to output video
    out.write(frame)

cap.release()
out.release()
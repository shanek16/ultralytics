import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.video  import VideoSink, VideoInfo
from sector import risk_area

window = 5
constant = 40
flow = 'sector' # arrow
current_file_path = os.path.dirname(os.path.abspath(__file__))
SOURCE_VIDEO_PATH = current_file_path + "/../../data/safety/video/fork4_1min.mp4"
TARGET_VIDEO_PATH = current_file_path + f"/../runs/warn/BoTSORT_fork4_window{window}_{flow}_x{constant}.mp4"

# Initialize YOLOv8 object detector
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
model = YOLO(current_file_path + "/../runs/detect/train4/weights/best.pt")

box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 1,
    text_scale = 0.5
)

# Initialize variables for previous frame and previous tracks
prev_frame = None
prev_tracks = None
xyxy_buffer = np.zeros((2,100), dtype=object)
v_buffer = []

# mean averge filter
def mean_last_rows(lst, window=5):
    n = min(len(lst), window)
    last_n_rows = np.array(lst[-n:])
    return np.mean(last_n_rows, axis=0)
# Savitzky-Golay filter for polynomial fitting
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for result in model.track(source = SOURCE_VIDEO_PATH, stream = True, agnostic_nms = True, tracker = "botsort.yaml"):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f'#{tracker_id} {model.model.names[class_id]}{confidence:0.2f}'
                for xyxy, mask, confidence,  class_id, tracker_id
                in detections 
            ]
        
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)

            # Calculate velocities and directions of tracked objects
            if prev_frame is not None and prev_tracks is not None:
                # Get position of object in current frame
                for xyxy, mask, confidence,  class_id, tracker_id in detections:
                    if class_id == 1 or class_id == 2:
                        x1, y1, x2, y2 = xyxy
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        xyxy_buffer[0][tracker_id-1] = (cx,cy)

                # Get position of object in previous frame
                for xyxy, mask, confidence,  class_id, tracker_id in prev_tracks:
                    if class_id == 1 or class_id == 2:
                        px1, py1, px2, py2 = xyxy
                        pcx = (px1 + px2) / 2
                        pcy = (py1 + py2) / 2
                        xyxy_buffer[1][tracker_id-1] = (pcx,pcy)

                for prev_point, current_point in zip(xyxy_buffer[1], xyxy_buffer[0]):
                    if prev_point != 0 and current_point !=0:
                        # Calculate velocity vector of object
                        vx = current_point[0] - prev_point[0]
                        vy = current_point[1] - prev_point[1]
                        v_buffer.append([vx,vy])
                        if len(v_buffer) > window:
                            v_buffer.pop(0)
                        mvx, mvy = mean_last_rows(v_buffer, window=window)
                        pt1x = int(prev_point[0])
                        pt1y = int(prev_point[1])
                        # pt2x = pt1x + constant*int(mvx)
                        # pt2y = pt1y + constant*int(mvy)
                        # frame = cv2.arrowedLine(frame, pt1=(pt1x, pt1y), pt2=(pt2x, pt2y), color=(0,0,255), thickness=2)
                        r = int(constant*np.sqrt(mvx**2 + mvy**2))
                        direction = np.arctan2(mvy, mvx) * 180 / np.pi
                        frame = risk_area(frame, center=(pt1x, pt1y), r=r, theta=direction)

        # Display frame with tracks and count
        # cv2.imshow("Frame", frame)

        # Update previous frame and previous tracks variables
        prev_frame = frame.copy()
        prev_tracks = detections

        sink.write_frame(frame)
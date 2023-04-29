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
buffer = np.zeros((4,100), dtype=object)
# coloumn
#------------------
# class_id
# current_[cx,cy]
# previous [cx,cy]
# mean [vx,vy]
#-------------------
v_buffer = []
prev_tracker_id = 0

def L2(p1,p2):
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        p1x,p1y = p1.flatten()
        p2x,p2y = p2.flatten()
        return np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)
    else: # either point is 0
        return 100

def risk_plot(frame, flow, buffer, mv, tracker_id, constant):
                        pt1x = int(buffer[2][tracker_id-1][0]) # prev_posx
                        pt1y = int(buffer[2][tracker_id-1][1]) # prev_posy
                        mvx = mv[tracker_id-1][0]
                        mvy = mv[tracker_id-1][1]
                        if flow == 'sector':
                            r = int(np.linalg.norm(mv[tracker_id-1]))
                            direction = np.arctan2(mvy, mvx) * 180 / np.pi
                            frame = risk_area(frame, center=(pt1x, pt1y), r=constant*r, theta=direction)
                        else:
                            pt2x = pt1x + constant*int(mvx)
                            pt2y = pt1y + constant*int(mvy)
                            frame = cv2.arrowedLine(frame, pt1=(pt1x, pt1y), pt2=(pt2x, pt2y), color=(0,0,255), thickness=2)

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for result in model.track(source = SOURCE_VIDEO_PATH, stream = True, agnostic_nms = True, tracker = "botsort.yaml"):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f'#{tracker_id} {model.model.names[class_id]}{confidence:0.2f}'
                for _, _, confidence,  class_id, tracker_id
                in detections 
            ]
        
            frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)

            # if new tracker_id: input class info into buffer[0]
            max_tracker_id = max(detections.tracker_id)
            for tracker_id in range(prev_tracker_id, max_tracker_id): # prev~ max_id-1
                buffer[0][tracker_id] = detections.class_id[tracker_id]
            prev_tracker_id = max_tracker_id

            # Calculate velocities and directions of tracked objects
            if prev_tracks is not None: # and prev_frame is not None and 
                # Get position of object in current frame
                for xyxy, mask, confidence, class_id, tracker_id in detections:
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    buffer[1][tracker_id-1] = np.array([cx,cy]) # current cx,cy
                    buffer[3][tracker_id-1] = detections.xyxy[0] # current xyxy
                
                v = buffer[1] - buffer[2]
                v_buffer.append(v)
                if len(v_buffer) > window:
                    v_buffer.pop(0)
                # mean averge filter
                if len(v_buffer) > 1:
                    mv = np.mean(np.array(v_buffer), axis=1) # resulting np.array of size 100
                else:
                    mv = v
                # Savitzky-Golay filter for polynomial fitting
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
                
                detected_forklift_tracker_id =[]
                detected_person_tracker_id = []
                #            0           1          2        3      
                # names: ['ladder', 'fork_lift', 'person','head'] 
                for xyxy, mask, confidence, class_id, tracker_id in detections:
                    if class_id == 1: # forklift
                        detected_forklift_tracker_id.append(tracker_id)
                    if class_id == 2: # person
                        detected_person_tracker_id.append(tracker_id)
                
                for tracker_id in detected_forklift_tracker_id:
                    risk_plot(frame, flow, buffer, mv, tracker_id, constant)


                for forklift_id in detected_forklift_tracker_id:
                    forklift_x1 = buffer[3][forklift_id-1][0]
                    forklift_x2 = buffer[3][forklift_id-1][2]
                    forklift_y1 = buffer[3][forklift_id-1][1]
                    forklift_y2 = buffer[3][forklift_id-1][3]
                    for person_id in detected_person_tracker_id:
                    # if person is not driver:
                        person_cx = buffer[1][person_id-1][0]
                        person_cy = buffer[1][person_id-1][1]
                        if person_cx < forklift_x1 or\
                            person_cx > forklift_x2 or\
                            person_cy < forklift_y1 or\
                            person_cy > forklift_y2:
                            risk_plot(frame, flow, buffer, mv, person_id, constant)
                buffer[2] = buffer[1]
            else:
                for xyxy, mask, confidence,  class_id, tracker_id in detections:
                    # if class_id == 1 or class_id == 2:
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    buffer[2][tracker_id-1] = np.array([cx,cy]) # previous row
        # Display frame with tracks and count
        # cv2.imshow("Frame", frame)

        # Update previous frame and previous tracks variables
        # prev_frame = frame.copy()
        prev_tracks = detections

        sink.write_frame(frame)
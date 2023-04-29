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
prev_tracks = None
pos_buffer = np.full((window,100), fill_value=None, dtype=object)
box_buffer = np.full(100, fill_value=None, dtype=object)
mv_buffer = np.full(100, fill_value=None, dtype=object)
# prev_tracker_id = 0

def L2(p1,p2):
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        p1x,p1y = p1.flatten()
        p2x,p2y = p2.flatten()
        return np.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)
    else: # either point is 0
        return 100

# Savitzky-Golay filter for polynomial fitting
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html\
def mean_average_filter(pos_buffer, mv_buffer, i):
    # if current pos is not None
    for tracker_id in range(np.shape(pos_buffer)[1]):
        if pos_buffer[i][tracker_id] is not None:
            # compute mv of tracker_id
            arr = pos_buffer[:, tracker_id]
            # Create a boolean mask to identify the [x,y] elements
            mask = [type(a) == np.ndarray and len(a) == 2 for a in arr]
            # Use the mask to index into the original array and select only the [x,y] elements
            new_arr = arr[mask]
            if len(new_arr) > 1:
                # Compute the differences between successive elements
                v_buffer = np.diff(new_arr, axis=0)
                mv_buffer[tracker_id] = np.mean(v_buffer)
    return mv_buffer  

def risk_plot(frame, flow, current_pos, mv_buffer, tracker_id, constant):
    pt1x = int(current_pos[tracker_id-1][0]) # prev_posx
    pt1y = int(current_pos[tracker_id-1][1]) # prev_posy
    if mv_buffer[tracker_id-1] is not None:
        mvx = mv_buffer[tracker_id-1][0]
        mvy = mv_buffer[tracker_id-1][1]
        if flow == 'sector':
            r = int(np.linalg.norm(mv_buffer[tracker_id-1]))
            direction = np.arctan2(mvy, mvx) * 180 / np.pi
            frame = risk_area(frame, center=(pt1x, pt1y), r=constant*r, theta=direction)
        else:
            pt2x = pt1x + constant*int(mvx)
            pt2y = pt1y + constant*int(mvy)
            frame = cv2.arrowedLine(frame, pt1=(pt1x, pt1y), pt2=(pt2x, pt2y), color=(0,0,255), thickness=2)
    return frame

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    i = 0
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

            # Calculate velocities and directions of tracked objects
            if prev_tracks is not None:
                # Get position of object in current frame
                for xyxy, mask, confidence, class_id, tracker_id in detections:
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    i = min(i, window-1)
                    pos_buffer[i][tracker_id-1] = np.array([cx,cy]) # current cx,cy
                    box_buffer[tracker_id-1] = detections.xyxy[0] # current xyxy
                
                # mean averge filter
                mv_buffer = mean_average_filter(pos_buffer, mv_buffer, i)

                detected_forklift_tracker_id =[]
                detected_person_tracker_id = []
                #            0           1          2        3      
                # names: ['ladder', 'fork_lift', 'person','head'] 
                for xyxy, mask, confidence, class_id, tracker_id in detections:
                    if class_id == 1: # forklift
                        detected_forklift_tracker_id.append(tracker_id)
                    if class_id == 2: # person
                        detected_person_tracker_id.append(tracker_id)
                
                current_pos = pos_buffer[i]
                for tracker_id in detected_forklift_tracker_id:
                    frame = risk_plot(frame, flow, current_pos, mv_buffer, tracker_id, constant*2)


                for forklift_id in detected_forklift_tracker_id:
                    forklift_x1 = box_buffer[forklift_id-1][0]
                    forklift_x2 = box_buffer[forklift_id-1][2]
                    forklift_y1 = box_buffer[forklift_id-1][1]
                    forklift_y2 = box_buffer[forklift_id-1][3]
                    for person_id in detected_person_tracker_id:
                    # if person is not driver:
                        person_cx = current_pos[person_id-1][0]
                        person_cy = current_pos[person_id-1][1]
                        if person_cx < forklift_x1 or\
                            person_cx > forklift_x2 or\
                            person_cy < forklift_y1 or\
                            person_cy > forklift_y2:
                            frame = risk_plot(frame, flow, current_pos, mv_buffer, person_id, constant)
                if i == window -1: # if buffer is full
                    pos_buffer[:-1, :] = pos_buffer[1:, :]
                    pos_buffer[-1, :] = None
                mv_buffer = np.full(100, None, dtype=object)
                box_buffer = np.full(100, None, dtype=object)
            else:
                for xyxy, mask, confidence,  class_id, tracker_id in detections:
                    # if class_id == 1 or class_id == 2:
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    pos_buffer[i][tracker_id-1] = np.array([cx,cy]) # previous row
                    prev_tracks = detections
            i=i+1
        # Display frame with tracks and count
        # cv2.imshow("Frame", frame)
        
        sink.write_frame(frame)
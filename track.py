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
EXPLAIN_VIDEO_PATH = current_file_path + f"/../runs/warn/Explain_BoTSORT_fork4_window{window}_{flow}_x{constant}.mp4"
WARNING_VIDEO_PATH = current_file_path + f"/../runs/warn/Warning_BoTSORT_fork4_window{window}_{flow}_x{constant}.mp4"
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
white_img = 255*np.ones((video_info.height, video_info.width, 3), dtype=np.uint8)

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

def warning(frame, xyxy, color=(0, 0, 255), alpha = 0.2):
    overlay = frame.copy()
    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    overlay = cv2.rectangle(overlay, p1, p2, color, -1, lineType=cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return frame

# explainable_video = VideoSink(EXPLAIN_VIDEO_PATH, video_info)
# warning_video = VideoSink(WARNING_VIDEO_PATH, video_info)

with VideoSink(EXPLAIN_VIDEO_PATH, video_info) as explainable_video:
    with VideoSink(WARNING_VIDEO_PATH, video_info) as warning_video:
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
                explainable_frame = frame.copy()
                warning_frame = frame.copy()
                # Calculate velocities and directions of tracked objects
                if prev_tracks is not None:
                    # Get position of object in current frame
                    for xyxy, mask, confidence, class_id, tracker_id in detections:
                        x1, y1, x2, y2 = xyxy
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        i = min(i, window-1)
                        pos_buffer[i][tracker_id-1] = np.array([cx,cy]) # current cx,cy
                        box_buffer[tracker_id-1] = xyxy # current xyxy, [0] for unpacking
                    
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
                    for forklift_id in detected_forklift_tracker_id:
                        forklift_frame = risk_plot(white_img, flow, current_pos, mv_buffer, forklift_id, constant*2)
                        explainable_frame = risk_plot(explainable_frame, flow, current_pos, mv_buffer, forklift_id, constant*2)
                        forklift_red_mask = cv2.inRange(forklift_frame, (153, 153, 255), (153, 153, 255))
                        forklift_orange_mask = cv2.inRange(forklift_frame, (153, 219, 255), (153, 219, 255))
                        forklift_yellow_mask = cv2.inRange(forklift_frame, (153, 255, 255), (153, 255, 255))
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
                                person_frame = risk_plot(white_img, flow, current_pos, mv_buffer, person_id, constant)
                                explainable_frame = risk_plot(explainable_frame, flow, current_pos, mv_buffer, person_id, constant)
                                person_red_mask = cv2.inRange(person_frame,(153, 153, 255), (153, 153, 255))
                                person_orange_mask = cv2.inRange(person_frame, (153, 219, 255), (153, 219, 255))
                                person_yellow_mask = cv2.inRange(person_frame, (153, 255, 255), (153, 255, 255))
                                
                                # xyxy for warning frame
                                index_forklift_id = np.where(detections.tracker_id == forklift_id)[0][0]
                                index_person_id = np.where(detections.tracker_id == person_id)[0][0]
                                x1 = min(detections.xyxy[index_forklift_id][0], detections.xyxy[index_person_id][0])
                                x2 = max(detections.xyxy[index_forklift_id][2], detections.xyxy[index_person_id][2])
                                y1 = min(detections.xyxy[index_forklift_id][1], detections.xyxy[index_person_id][1])
                                y2 = max(detections.xyxy[index_forklift_id][3], detections.xyxy[index_person_id][3])
                                # Check for overlapping sectors
                                if cv2.countNonZero(cv2.bitwise_and(forklift_red_mask, person_red_mask)) > 0:
                                    print('emergency')
                                    warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,0,255))
                                    # input()
                                elif cv2.countNonZero(cv2.bitwise_and(forklift_orange_mask, person_orange_mask)) > 0:
                                    print('danger')
                                    warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,165,255))
                                    # input()                                    
                                elif cv2.countNonZero(cv2.bitwise_and(forklift_yellow_mask, person_yellow_mask)) > 0:
                                    print('warning')
                                    warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,255,255))
                                    # input()

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
            explainable_video.write_frame(explainable_frame)
            warning_video.write_frame(warning_frame)

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.video  import VideoSink, VideoInfo
from sector import risk_area
import torch
# torch.cuda.set_device(0)

from midas.model_loader import default_models, load_model
# from jtop import jtop
# import threading
import tensorrt as trt
import pycuda.driver as cuda
trt.init_libnvinfer_plugins(None, "")
import argparse

parser = argparse.ArgumentParser(description='receive input video file name')
parser.add_argument('--input', type=str, required=True , help='input video file in abs path')
args = parser.parse_args()

cuda.init()
# Get the default CUDA device
cuda_device = cuda.Device(0)
# Create a context for the device
context = cuda_device.make_context()

window = 10
constant = 40
flow = 'sector' # arrow
current_file_path = os.path.dirname(os.path.abspath(__file__))
file_name = args.input
# file_name = 'fork_container_1min'

# server
# SOURCE_VIDEO_PATH = current_file_path + f"/../../data/safety/video/{file_name}.mp4"
# SOURCE_VIDEO_PATH = current_file_path + f"/../../data/safety/video_official/{file_name}.mp4

# jetson
# # SOURCE_VIDEO_PATH = current_file_path + f"/../../../../media/shane/44B4-A589/video/{file_name}.mp4"
# EXPLAIN_VIDEO_PATH = current_file_path + f"/../runs/warn/Explain_BoTSORT_{file_name}_window{window}_{flow}_x{constant}.mp4"
# WARNING_VIDEO_PATH = current_file_path + f"/../runs/warn/Warning_BoTSORT_{file_name}_window{window}_{flow}_x{constant}.mp4"
# DEBUG_VIDEO_PATH = current_file_path + f"/../runs/warn/DEBUG.mp4"

# Docker
SOURCE_VIDEO_PATH = current_file_path + f"/../data/safety/video/{file_name}.mp4" # server
EXPLAIN_VIDEO_PATH = current_file_path + f"/runs/warn/Explain_engine_{file_name}.mp4"
WARNING_VIDEO_PATH = current_file_path + f"/runs/warn/Warning_engine_{file_name}.mp4"
DEBUG_VIDEO_PATH = current_file_path + f"/runs/warn/DEBUG_engine_{file_name}.mp4"

# Initialize YOLOv8 object detector
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# model = YOLO("/home/shane/Project/weights/detect/engine/Lbest_simplify.engine") # jetson
# model = YOLO("/home/swkim/Project/yolov8_tracking/runs/detect/train3/weights/best.engine") # server
model = YOLO(current_file_path + "/../weights/detect/YOLOL.engine")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load MiDAS engine
logger = trt.Logger(trt.Logger.WARNING) #VERBOSE
runtime = trt.Runtime(logger)
# jetson
# with open("/media/shane/44B4-A589/weights/depth/engine/midas_dpt_hybrid_simplified.engine", "rb") as f:
# server
# with open("/home/swkim/Project/weights/depth/engine/midas_dpt_hybrid_simplified.engine", "rb") as f:
# ssw
with open("/workspace/weights/depth/midas_dpt_hybrid_simplified.engine", "rb") as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)
context.push()
try:
    midas = engine.create_execution_context()
except Exception as e:
    print(f"Error creating MiDaS execution context: {e}")
    context.pop()
    exit()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

box_annotator = sv.BoxAnnotator(
    thickness = 2,
    text_thickness = 1,
    text_scale = 0.5
)

# Initialize variables for previous frame and previous tracks
prev_tracks = None
pos_buffer = np.full((window,300), fill_value=None, dtype=object)
box_buffer = np.full(300, fill_value=None, dtype=object)
mv_buffer = np.full(300, fill_value=None, dtype=object) # seems like mean velocity buffer
height = video_info.height
width = video_info.width
white_img = 255*np.ones((height, width, 3), dtype=np.uint8)
model_names = ['ladder', 'fork_lift', 'person','head']

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
    """
    Apply a warning bounding box overlay to a given frame.

    Parameters:
    - frame: The input frame to apply the warning overlay to.
    - xyxy: A tuple representing the bounding box coordinates (x1, y1, x2, y2).
    - color: Optional. The color of the warning overlay in BGR format. Default is (0, 0, 255) (red).
    - alpha: Optional. The transparency of the warning overlay. Default is 0.2.

    Returns:
    - The frame with the warning bounding box overlay applied.
    """
    overlay = frame.copy()
    p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    overlay = cv2.rectangle(overlay, p1, p2, color, -1, lineType=cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return frame

def compute_median_distance(depth_map):
    h, w = depth_map.shape
    center_x, center_y = w // 2, h // 2

    # Crop center (50%)
    cropped_h, cropped_w = h // 2, w // 2
    start_x, start_y = center_x - cropped_w // 2, center_y - cropped_h // 2

    cropped_image = depth_map[start_y:start_y+cropped_h, start_x:start_x+cropped_w]

    # Sample 100 pixels
    sampled_pixels = np.random.choice(cropped_image.ravel(), size=100)
    # why not just take median of cropped_image.ravel()? -> maybe because of inference speed.
    return np.median(sampled_pixels)

def is_point_in_rect(point, rect):
    """
    Check if a point (x, y) is within a rectangle defined by a NumPy array [x1, y1, x2, y2].
    """
    (x1, y1, x2, y2) = rect
    x, y = point
    return x1 < x < x2 and y1 < y < y2

def is_point_in_sum_set(point, rects):
    """
    Check if a point (x, y) is within the sum-set of rectangles given as a list of NumPy arrays.
    """
    return any(is_point_in_rect(point, rect) for rect in rects)

# def power_monitoring(samples):
#     with jtop() as jetson:
#         while not exit_flag:
#             power = jetson.power['tot']['power']
#             samples.append(power)
#             total_memory = jetson.memory['RAM']['used'] + jetson.memory['SWAP']['used']
#             memory_samples.append(total_memory)
#             time.sleep(1)  # sample every second

# exit_flag = False
# samples = []
# memory_samples = []

# # Start the power monitoring thread
# monitor_thread = threading.Thread(target=power_monitoring, args=(samples,))
# monitor_thread.start()

# out = cv2.VideoWriter(DEBUG_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (2 * width, 2 * height))
video_info.codec = "MJPG"  # or "XVID"
try:
    with VideoSink(WARNING_VIDEO_PATH, video_info) as warning_video:
        with VideoSink(EXPLAIN_VIDEO_PATH, video_info) as explainable_video:

            i = 0
            # FPS and Throughput measurement
            start_time = time.time()
            frame_count = 0
            # total_data_processed_MB = 0

            # Set up buffers
            input_shape = (1, 3, 384, 672)  
            output_shape = (1, 1, 384, 672)  
            h_input = np.empty(input_shape, dtype=np.float32)
            h_output = np.empty(output_shape, dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            bindings = [int(d_input), int(d_output)]
            for result in model.track(source = SOURCE_VIDEO_PATH, stream = True, agnostic_nms = True, tracker = "botsort.yaml"):
                frame = result.orig_img
                # print(np.shape(frame)) # (1080, 1920, 3)
                frame_count += 1
                # frame_data_MB = frame.shape[0] * frame.shape[1] * frame.shape[2] * 8 / (8 * 1024 * 1024)  # 8 bits per channel
                # total_data_processed_MB += frame_data_MB
                if result.boxes.id is not None:
                    # Depth inference
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # input_batch = transform(img).to(device)
                    input_batch = transform(img).to(device).cpu().numpy()
                    # print(np.shape(input_batch)) # torch.Size([1, 3, 384, 672])

                    # Set input data
                    np.copyto(h_input, input_batch)
                    cuda.memcpy_htod(d_input, h_input)
                    
                    # Execute the model
                    midas.execute_v2(bindings=bindings)
                    
                    # Retrieve the output data
                    cuda.memcpy_dtoh(h_output, d_output)
                    prediction = torch.from_numpy(h_output)
                    # print('shape of predition: ', prediction.shape) # torch.Size([1, 1, 384, 672])
                    prediction = torch.nn.functional.interpolate(
                        prediction,
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    # print(prediction.shape) # torch.Size([1080, 1920])
                    output = prediction.cpu().numpy()
                    # t_end = time.time()
                    # print("depth estimation inference time : ", t_end - t_start)
                    
                    # Detection & Tracking inference
                    detections = sv.Detections.from_yolov8(result)
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    labels = [
                        f'#{tracker_id} {model_names[class_id]}{confidence:0.2f}'
                        for _, _, confidence,  class_id, tracker_id
                        in detections 
                    ]
                
                    frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
                    explainable_frame = frame.copy()
                    warning_frame = frame.copy()
                    forklift_yellow_mask = np.zeros((height, width), dtype=np.uint8)
                    person_yellow_mask = np.zeros((height, width), dtype=np.uint8)
                    # Calculate velocities and directions of tracked objects
                    if prev_tracks is not None:
                        detected_forklift_tracker_id =[]
                        detected_person_tracker_id = []
                        forklift_xyxy = []
                        # Get position of object in current frame
                        for xyxy, mask, confidence, class_id, tracker_id in detections:
                            #            0           1          2        3      
                            # names: ['ladder', 'fork_lift', 'person','head'] 
                            if class_id == 1: # forklift
                                detected_forklift_tracker_id.append(tracker_id)
                                forklift_xyxy.append(xyxy) #xyxy: numpy array
                            if class_id == 2: # person
                                detected_person_tracker_id.append(tracker_id)
                            x1, y1, x2, y2 = xyxy
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            i = min(i, window-1)
                            pos_buffer[i][tracker_id-1] = np.array([cx,cy]) # current cx,cy
                            box_buffer[tracker_id-1] = xyxy # current xyxy, [0] for unpacking

                        # mean averge filter
                        mv_buffer = mean_average_filter(pos_buffer, mv_buffer, i)
                        
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
                            depth_forklift = compute_median_distance(output[max(round(forklift_y1), 0):round(forklift_y2),max(round(forklift_x1), 0):round(forklift_x2)])
                            cv2.putText(explainable_frame, str(round(depth_forklift, 2)), (round(forklift_x1) + 5, round(forklift_y2) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)

                            for person_id in detected_person_tracker_id:
                                person_cx = int(current_pos[person_id-1][0])
                                person_cy = int(current_pos[person_id-1][1])
                                # if person is not driver:
                                if not is_point_in_sum_set((person_cx, person_cy), forklift_xyxy):
                                # if person_cx < forklift_x1 or\
                                #     person_cx > forklift_x2 or\
                                #     person_cy < forklift_y1 or\
                                #     person_cy > forklift_y2:
                                    person_x1 = int(box_buffer[person_id-1][0])
                                    person_x2 = int(box_buffer[person_id-1][2])
                                    person_y1 = int(box_buffer[person_id-1][1])
                                    person_y2 = int(box_buffer[person_id-1][3])
                                    person_frame = risk_plot(white_img, flow, current_pos, mv_buffer, person_id, constant)
                                    explainable_frame = risk_plot(explainable_frame, flow, current_pos, mv_buffer, person_id, constant)
                                    person_red_mask = cv2.inRange(person_frame,(153, 153, 255), (153, 153, 255))
                                    person_red_mask[person_y1:person_y2, person_x1:person_x2] = 255
                                    person_orange_mask = cv2.inRange(person_frame, (153, 219, 255), (153, 219, 255))
                                    person_orange_mask[person_y1:person_y2, person_x1:person_x2] = 255
                                    person_yellow_mask = cv2.inRange(person_frame, (153, 255, 255), (153, 255, 255))
                                    person_yellow_mask[person_y1:person_y2, person_x1:person_x2] = 255
                                    depth_person = compute_median_distance(output[max(round(person_y1), 0):round(person_y2),max(round(person_x1), 0):round(person_x2)])
                                    # cv2.putText(explainable_frame, str(round(depth_person, 2)), (round(person_x1) + 5, round(person_y2) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)
                                    # Check for overlapping sectors
                                    if abs(depth_forklift - depth_person) < 200.0:
                                        emergency_mask = cv2.bitwise_and(forklift_red_mask, person_red_mask)
                                        danger_mask = cv2.bitwise_and(forklift_orange_mask, person_orange_mask)
                                        warning_mask = cv2.bitwise_and(forklift_yellow_mask, person_yellow_mask)
                                        # xyxy for warning frame
                                        index_forklift_id = np.where(detections.tracker_id == forklift_id)[0][0]
                                        index_person_id = np.where(detections.tracker_id == person_id)[0][0]
                                        x1 = min(detections.xyxy[index_forklift_id][0], detections.xyxy[index_person_id][0])
                                        x2 = max(detections.xyxy[index_forklift_id][2], detections.xyxy[index_person_id][2])
                                        y1 = min(detections.xyxy[index_forklift_id][1], detections.xyxy[index_person_id][1])
                                        y2 = max(detections.xyxy[index_forklift_id][3], detections.xyxy[index_person_id][3])
                                        if cv2.countNonZero(emergency_mask) > 0:
                                            warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,0,255))
                                        elif cv2.countNonZero(danger_mask) > 0:
                                            warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,165,255))
                                        elif cv2.countNonZero(warning_mask) > 0:
                                            warning_frame = warning(warning_frame, [x1,y1,x2,y2], color=(0,255,255))

                        if i == window -1: # if buffer is full
                            pos_buffer[:-1, :] = pos_buffer[1:, :]
                            pos_buffer[-1, :] = None
                        mv_buffer = np.full(300, None, dtype=object)
                        box_buffer = np.full(300, None, dtype=object)
                    else:
                        for xyxy, mask, confidence,  class_id, tracker_id in detections:
                            # if class_id == 1 or class_id == 2:
                            x1, y1, x2, y2 = xyxy
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            pos_buffer[i][tracker_id-1] = np.array([cx,cy]) # previous row
                            prev_tracks = detections
                    i=i+1
                else:
                    warning_frame = frame.copy()
                    explainable_frame = frame.copy()
                # Display frame with tracks and count
                # cv2.imshow("Frame", frame)
                # cv2.imshow("Explainable Frame", explainable_frame)
                # cv2.waitKey(1)
                # cv2.imshow("Warning frame", warning_frame)
                
                # save frames
                explainable_video.write_frame(explainable_frame)
                warning_video.write_frame(warning_frame)

                # create an empty canvas with twice the width and height of the frames(to see 4 frames in one canvas: for debugging)
                # canvas = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

                # place each frame in the appropriate quadrant of the canvas
                # canvas[:height, :width] = explainable_frame
                # canvas[:height, width:] = warning_frame
                # canvas[height:, :width] = cv2.cvtColor(forklift_yellow_mask, cv2.COLOR_GRAY2BGR)
                # canvas[height:, width:] = cv2.cvtColor(person_yellow_mask, cv2.COLOR_GRAY2BGR)

                # write the canvas to the output video file
                # out.write(canvas)
            # Calculate FPS and Throughput
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            # throughput_MB_per_s = total_data_processed_MB / elapsed_time
            print(f"Average FPS: {fps:.2f}")
            # print(f"Throughput: {throughput_MB_per_s:.2f} MB/s")

except Exception as e:
    print(f"Error during processing: {e}")
# exit_flag = True
# monitor_thread.join()

# # Calculate the total power consumed
# total_power_consumed = sum(samples) - samples[0]  # subtracting the initial power reading
# print(f"Total power consumed: {total_power_consumed:.2f} mW")
# # Calculate the max and min memory used
# memory_usage_difference = max(memory_samples) - min(memory_samples)
# print(f"Memory usage difference (max - min): {memory_usage_difference:.2f} KB")
finally:
    if context:
        context.pop()
# out.release()

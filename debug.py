# import cv2

video_path = "/workspace/data/safety/video/container_1min.mp4"

try:
    with open(video_path, 'rb') as video_file:
        # Do something with the video file, e.g., read a few bytes.
        data = video_file.read(10)
        print(f"Read {len(data)} bytes from the video file.")
except Exception as e:
    print(f"Error: {str(e)}")

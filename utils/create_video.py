import numpy as np
import cv2

path = "test.mp4"
video_capture = cv2.VideoCapture(path)
frame_rate = 30
width, height = 3840, 2160
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter("output.avi", fourcc, frame_rate, (width, height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.rectangle(frame, (100, 100), (1024, 1024), color=(255, 0, 0), thickness=20)
    video_writer.write(frame)

video_capture.release()
video_writer.release()

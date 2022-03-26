import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)
    cv2.imshow('object detection',  cv2.resize(image_np, (800, 600)))
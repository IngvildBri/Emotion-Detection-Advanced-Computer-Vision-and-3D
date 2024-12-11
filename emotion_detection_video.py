# Import dependencies
import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


# Path to weights from the trained model on faces and YOLOv5 model
model_path = ''
model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)


# Connects to camera
cap = cv2.VideoCapture(0)


# Program runs for real-time object detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    print("Frame captured.")
    
    results = model_detect(frame) # Object detection model

    results.render() # Bounding boxes and labels from YOLOv5

    cv2.imshow('Face detection with YOLOYv5', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
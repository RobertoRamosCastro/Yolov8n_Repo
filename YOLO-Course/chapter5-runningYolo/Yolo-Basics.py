from ultralytics import YOLO
import cv2

model = YOLO(r'D:\YOLO-Course\chapter5-runningYolo\YOLO-Weights\yolov8n.pt')
results = model(r"D:\YOLO-Course\chapter5-runningYolo\Images\3.png", show=True)

cv2.waitKey(0)


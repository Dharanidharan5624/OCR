import sys, os
from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")
image = cv2.imread("/home/ubuntu/Documents/new_coin/2106589-1.jpg")
image = cv2.resize(image, (640, 640))
res = model.predict(image, device="cpu", conf=0.75, classes=0)[0]
for idx, box in enumerate(res.boxes):
    b = box.xyxy.cpu().numpy().squeeze()
    print("Box", idx, ":", b)

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
import torch
import cv2

model = YOLO("C:/Users/natt4/PycharmProjects/RoverMachineLearning/runs/detect/train3/weights/best.pt")

results = model.predict(source="0", show = "true")

print(results)
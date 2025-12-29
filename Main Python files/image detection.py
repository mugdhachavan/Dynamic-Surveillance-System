from ultralytics import YOLO
import cv2

model = YOLO('../YoloWeights/dynamic2.pt')
results = model("D:\PycharmProjects\MiniProject\images\cctv.jpg", show=True)
cv2.waitKey(0)
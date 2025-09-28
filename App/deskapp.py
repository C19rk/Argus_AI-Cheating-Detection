import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === Paths ===
base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(base_dir, 'runs', 'aidetection4', 'weights', 'best.pt')

# Load models
yolo = YOLO(yolo_model_path)

# === Webcam
rtsp_url = "rtsp://administrator:admin123@192.168.100.203:554/stream2"
# kung iba ang network: rtsp://<username>:<password>@<ip_addr>:554/stream#
# 1 para sa hd, 2 kung low quality
# kailangan ninyo maginstall ng Tapo app, magregister at gumawa pa na isa pang account para camera. dahil tatlo ang mga camera, siguro tatlo din ang mga account.

#cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = yolo.predict(source=frame, imgsz=640, conf=0.25, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw YOLO box
            color = (255, 0, 0) if label == "person" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the final frame
    cv2.imshow("Argus - AI Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

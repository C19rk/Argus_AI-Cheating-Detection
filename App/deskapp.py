import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
#import mediapipe as mp

# === CNN Pose Classifier ===
#class PoseCNN(torch.nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.net = torch.nn.Sequential(
#            torch.nn.Linear(99, 128),
#            torch.nn.ReLU(),
#            torch.nn.Linear(128, 64),
#            torch.nn.ReLU(),
#            torch.nn.Linear(64, num_classes)
#        )
#
#    def forward(self, x):
#        return self.net(x)
#
## === Pose Estimator with Drawing ===
#mp_pose = mp.solutions.pose
#mp_drawing = mp.solutions.drawing_utils
#pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
#
#def extract_pose_keypoints_and_draw(frame):
#    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#    result = pose_detector.process(frame_rgb)
#
#    if result.pose_landmarks:
#        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#        keypoints = result.pose_landmarks.landmark
#        return [coord for lm in keypoints for coord in (lm.x, lm.y, lm.z)][:99]
#    return None

# === Paths ===
base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(base_dir, 'runs', 'aidetection2', 'weights', 'best.pt')
#pose_model_path = os.path.join(base_dir, 'model', 'pose_cnn.pt')
#labels_path = os.path.join(base_dir, 'labels.txt')

# Load labels
#with open(labels_path) as f:
#    pose_labels = f.read().splitlines()

# Load models
yolo = YOLO(yolo_model_path)
#pose_model = PoseCNN(num_classes=len(pose_labels))
#pose_model.load_state_dict(torch.load(pose_model_path))
#pose_model.eval()

# === Webcam
rtsp_url = "rtsp://administrator:admin123@192.168.100.203:554/stream2"
# kung iba ang network: rtsp://<username>:<password>@<ip_addr>:554/stream# 
# 1 para sa hd, 2 kung low quality
# kailangan ninyo maginstall ng Tapo app, magregister at gumawa pa na isa pang account para camera. dahil tatlo ang mga camera, siguro tatlo din ang mga account.

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Always draw pose skeleton on full frame
    pose_vec = extract_pose_keypoints_and_draw(frame)

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

    # Run pose classification (if pose was detected)
#    if pose_vec is not None:
#        with torch.no_grad():
#            inp = torch.tensor(pose_vec).unsqueeze(0)
#            output = pose_model(inp)
#            pred_idx = torch.argmax(output, dim=1).item()
#            confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
#            pose_label = pose_labels[pred_idx]
#
#            if confidence > 0.5:
#                cv2.putText(frame, f"{pose_label} ({confidence:.2f})", (10, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the final frame
    cv2.imshow("Argus - AI Cheating Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import os

# Get absolute path to the current script directory (App folder)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to the YOLO model inside the App folder
model_path = os.path.join(base_dir, 'yolov10n.pt')

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    results.render()

    cv2.imshow('Argus - AI Cheating Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

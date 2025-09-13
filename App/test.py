import cv2

cap = cv2.VideoCapture(0)  # use 0 for default webcam

if not cap.isOpened():
    print("❌ Could not open webcam")
else:
    print("✅ Webcam opened successfully")

cap.release()

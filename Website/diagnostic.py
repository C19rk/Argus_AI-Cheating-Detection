from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import threading

app = Flask(__name__)

# === Configuration ===
RTSP_URL = "rtsp://admin3:admin123@192.168.100.203:5542"
yolo_model_path = "../App/runs/aidetection2/weights/best.pt"

# === Global Variables ===
current_frame = None
frame_lock = threading.Lock()
camera_status = "Initializing..."
running = True

# Load YOLO
try:
    yolo = YOLO(yolo_model_path)
    print("✓ YOLO model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load YOLO model: {e}")
    yolo = None

def camera_thread():
    global current_frame, camera_status, running
    
    print(f"Starting camera thread with URL: {RTSP_URL}")
    
    # Try different connection methods
    connection_methods = [
        {"backend": cv2.CAP_FFMPEG, "name": "FFMPEG"},
        {"backend": cv2.CAP_ANY, "name": "Default"},
    ]
    
    cap = None
    
    for method in connection_methods:
        try:
            print(f"Trying {method['name']} backend...")
            cap = cv2.VideoCapture(RTSP_URL, method["backend"])
            
            # Set properties
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # 15 seconds
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # 10 seconds
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                print(f"✓ Camera opened with {method['name']} backend")
                
                # Test reading a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✓ Successfully read test frame: {test_frame.shape}")
                    camera_status = "Connected"
                    break
                else:
                    print(f"✗ Could not read frame with {method['name']}")
                    cap.release()
                    cap = None
            else:
                print(f"✗ Could not open camera with {method['name']}")
                cap.release()
                cap = None
                
        except Exception as e:
            print(f"✗ Exception with {method['name']}: {e}")
            if cap:
                cap.release()
            cap = None
    
    if not cap or not cap.isOpened():
        camera_status = "Failed to connect"
        print("✗ All connection methods failed")
        return
    
    # Main processing loop
    consecutive_failures = 0
    max_failures = 5
    
    while running:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"Frame read failed (attempt {consecutive_failures})")
                
                if consecutive_failures >= max_failures:
                    camera_status = "Connection lost"
                    break
                    
                time.sleep(1)
                continue
            
            # Reset failure counter
            consecutive_failures = 0
            camera_status = "Active"
            
            # Process with YOLO if available
            if yolo is not None:
                try:
                    results = yolo.predict(frame, imgsz=640, conf=0.25, verbose=False)

                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                cls = int(box.cls[0].item())
                                label = yolo.names[cls]
                                conf = float(box.conf[0].item())
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                                color = (255, 0, 0) if l

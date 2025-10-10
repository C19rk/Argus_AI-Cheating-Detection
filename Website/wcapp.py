from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import logging
import time
import os
import numpy as np

# === Suppress OpenCV FFmpeg warnings ===
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

app = Flask(__name__)

# === Logging Setup ===
log_file = "detections.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# === Simulated GSM module ===
class GSM:
    def __init__(self, port=None, baudrate=None):
        self.port = port
        self.baudrate = baudrate
        print(f"[SIM] GSM module initialized on port {port} at {baudrate} baud")

    def send_sms(self, number, message):
        log_msg = f"[SIM] Sending SMS to {number}: {message}"
        print(log_msg)
        logging.info(log_msg)
        time.sleep(0.5)  # simulate delay

# === Initialize simulated GSM ===
gsm = GSM(port="COM3", baudrate=9600)

# === YOLO models ===
yolo_model_path = "../App/runs/aidetection7/weights/best.pt"
yolo = YOLO(yolo_model_path)
yolo_pose = YOLO("../App/yolo11n-pose.pt")  # Pose model

# === Camera sources ===
camera_sources = {
    "cam1": 0,
    "cam2": 0,
    "cam3": 0,
}

# === VideoCapture objects ===
caps = {name: cv2.VideoCapture(url) for name, url in camera_sources.items()}
for cap in caps.values():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Draw skeleton ===
def draw_skeleton(frame, keypoints, conf_threshold=0.3, color=(0, 255, 0), thickness=2):
    connections = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
        (5,6),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]
    for i,j in connections:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints.shape[1]==3:
                x1,y1,c1 = keypoints[i]
                x2,y2,c2 = keypoints[j]
                if c1<conf_threshold or c2<conf_threshold:
                    continue
            else:
                x1,y1 = keypoints[i]
                x2,y2 = keypoints[j]
            cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,thickness)
    for kp in keypoints:
        if keypoints.shape[1]==3:
            x,y,c = kp
            if c<conf_threshold: continue
        else:
            x,y = kp
        cv2.circle(frame,(int(x),int(y)),3,color,-1)

# === Alert cooldown and phone numbers ===
phone_numbers = ["+639XXXXXXXXX", "+639YYYYYYYYY"]  # add your numbers here
last_alert_time = {}
ALERT_COOLDOWN = 10  # seconds
suspicious_labels = ["phone","talking"]  # labels that trigger SMS

# === Video generator ===
def generate_frames(cam_name):
    cap = caps[cam_name]
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            continue

        # YOLO object detection
        results = yolo.predict(frame, imgsz=640, conf=0.50)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls]
                conf = float(box.conf[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

                color = (255,0,0) if label=="person" else (0,255,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

                log_msg = f"[{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]"
                print(log_msg)
                logging.info(log_msg)

                # Send SMS if suspicious and cooldown passed
                if label in suspicious_labels:
                    key = f"{cam_name}_{label}"
                    now = time.time()
                    if key not in last_alert_time or now - last_alert_time[key] > ALERT_COOLDOWN:
                        for number in phone_numbers:
                            gsm.send_sms(number,f"Alert! Detected {label} on {cam_name}")
                        last_alert_time[key] = now

        # === YOLO pose detection and head direction ===
        pose_results = yolo_pose.predict(frame, imgsz=640, conf=0.25)
        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                for person_kpts in r.keypoints.xy:
                    keypoints_np = person_kpts.cpu().numpy()
                    draw_skeleton(frame, keypoints_np)

                    # --- Head direction detection (normalized and stable) ---
                    def get_point(idx):
                        return keypoints_np[idx] if idx < len(keypoints_np) else None

                    nose = get_point(0)
                    left_eye = get_point(1)
                    right_eye = get_point(2)
                    left_shoulder = get_point(5)
                    right_shoulder = get_point(6)

                    if any(p is None for p in [nose, left_eye, right_eye, left_shoulder, right_shoulder]):
                        continue

                    # Extract x, y
                    nx, ny = nose[0], nose[1]
                    lex, ley = left_eye[0], left_eye[1]
                    rex, rey = right_eye[0], right_eye[1]
                    lsx, lsy = left_shoulder[0], left_shoulder[1]
                    rsx, rsy = right_shoulder[0], right_shoulder[1]

                    # Compute reference points
                    eye_mid_x = (lex + rex) / 2
                    eye_mid_y = (ley + rey) / 2
                    shoulder_mid_y = (lsy + rsy) / 2

                    # Normalize distances
                    eye_dist = max(abs(rex - lex), 1e-6)
                    offset_x = (nx - eye_mid_x) / eye_dist
                    offset_y = (ny - eye_mid_y) / eye_dist

                    # Thresholds (tweak if needed)
                    H_THRESH = 0.30   # left/right sensitivity
                    DOWN_THRESH = 0.45
                    UP_THRESH = -0.35

                    label = "facing_camera"

                    if offset_x > H_THRESH:
                        label = "looking_right"
                    elif offset_x < -H_THRESH:
                        label = "looking_left"
                    elif offset_y > DOWN_THRESH:
                        label = "looking_down"
                    elif offset_y < UP_THRESH:
                        label = "looking_up"
                    else:
                        label = "facing_camera"

                    # Draw label
                    x, y = int(nx), int(ny) - 10
                    cv2.putText(frame, label, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Log result
                    log_msg = f"[{cam_name}] Pose label: {label}"
                    print(log_msg)
                    logging.info(log_msg)

        ret, buffer = cv2.imencode('.jpg',frame)
        if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

# === Log streaming ===
def follow(logfile):
    logfile.seek(0,2)
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"

# === Flask routes ===
@app.route('/')
def index():
    template = """
    <html>
    <head>
        <title>Argus - AI Cheating Detection</title>
        <style>
            body { font-family: monospace; margin:0; padding:0; }
            .container { display: flex; height: 100vh; }
            .video { flex: 2; padding: 10px; }
            .log { flex: 1; padding: 10px; background-color: #f0f0f0; overflow-y: scroll; border-left: 2px solid #ccc; }
            img { width: 100%; height: auto; }
            .buttons { margin-bottom: 10px; }
            button { margin-right: 10px; padding: 8px 16px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video">
                <h2>Live Video</h2>
                <div class="buttons">
                    <button onclick="switchCam('cam1')">Camera 1</button>
                    <button onclick="switchCam('cam2')">Camera 2</button>
                    <button onclick="switchCam('cam3')">Camera 3</button>
                </div>
                <img id="videoStream" src="/video/cam1">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
            function switchCam(camName){ document.getElementById("videoStream").src="/video/"+camName; }
            var evtSource = new EventSource("/log_stream");
            var logDiv = document.getElementById("log");
            evtSource.onmessage = function(e){ logDiv.innerHTML+=e.data+"<br>"; logDiv.scrollTop=logDiv.scrollHeight; }
        </script>
    </body>
    </html>
    """
    return render_template_string(template)

@app.route('/video/<cam_name>')
def video(cam_name):
    if cam_name not in caps:
        return "Camera not found", 404
    return Response(generate_frames(cam_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/log_stream')
def log_stream():
    logfile = open(log_file,"r")
    return Response(follow(logfile),mimetype="text/event-stream")

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

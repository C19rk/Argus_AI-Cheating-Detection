from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import logging
import time
import os
import numpy as np
import threading

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

# Initialize simulated GSM
gsm = GSM(port="COM3", baudrate=9600)

# === Load YOLO models ===
yolo_model_path = "../App/runs/aidetection7/weights/best.pt"
yolo = YOLO(yolo_model_path)
yolo_pose = YOLO("../App/yolo11n-pose.pt")  # Pose model

# === Camera RTSP URLs (kept as requested) ===
camera_sources = {
    "cam1": "rtsp://admin1:admin123@192.168.100.201:554/stream2",
    "cam2": "rtsp://admin2:admin123@192.168.100.202:554/stream2",
    "cam3": "rtsp://admin3:admin123@192.168.100.203:554/stream2",
}

# === Camera VideoCapture objects ===
caps = {name: cv2.VideoCapture(url) for name, url in camera_sources.items()}
for cap in caps.values():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Draw skeleton ===
def draw_skeleton(frame, keypoints, conf_threshold=0.3, color=(0, 255, 0), thickness=2):
    """
    keypoints: numpy array shape (N,2) or (N,3) where last column is confidence (optional)
    """
    connections = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
        (5,6),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints.shape[1] == 3:
                x1, y1, c1 = keypoints[i]
                x2, y2, c2 = keypoints[j]
                if c1 < conf_threshold or c2 < conf_threshold:
                    continue
            else:
                x1, y1 = keypoints[i]
                x2, y2 = keypoints[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    for kp in keypoints:
        if keypoints.shape[1] == 3:
            x, y, c = kp
            if c < conf_threshold:
                continue
        else:
            x, y = kp
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

# === Alert cooldown and phone numbers ===
phone_numbers = ["+639XXXXXXXXX", "+639YYYYYYYYY"]  # add your numbers
last_alert_time = {}
ALERT_COOLDOWN = 10  # seconds
suspicious_labels = ["phone", "talking"]  # labels that trigger SMS

# === Video generator ===
def generate_frames(cam_name):
    cap = caps[cam_name]
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            # if RTSP hiccup, small sleep to avoid busy loop
            time.sleep(0.05)
            continue

        # === YOLO object detection ===
        try:
            results = yolo.predict(frame, imgsz=640, conf=0.50)
        except Exception as e:
            print(f"[{cam_name}] YOLO object detection error: {e}")
            logging.exception(e)
            results = []

        for r in results:
            # r.boxes could be empty
            for box in getattr(r, "boxes", []):
                try:
                    cls = int(box.cls[0].item())
                    label = yolo.names[cls]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                except Exception:
                    # fallback in case shapes differ
                    continue

                # Draw YOLO box
                color = (255, 0, 0) if label == "person" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Log detection (always)
                log_msg = f"[{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]"
                print(log_msg)
                logging.info(log_msg)

                # === Send SMS if suspicious and cooldown passed ===
                if label in suspicious_labels:
                    key = f"{cam_name}_{label}"
                    now = time.time()
                    if key not in last_alert_time or now - last_alert_time[key] > ALERT_COOLDOWN:
                        for number in phone_numbers:
                            try:
                                gsm.send_sms(number, f"Alert! Detected {label} on {cam_name}")
                            except Exception as e:
                                logging.exception(f"Failed to send SMS: {e}")
                        last_alert_time[key] = now

        # === YOLO pose detection + head direction ===
        try:
            pose_results = yolo_pose.predict(frame, imgsz=640, conf=0.25)
        except Exception as e:
            print(f"[{cam_name}] YOLO pose detection error: {e}")
            logging.exception(e)
            pose_results = []

        for r in pose_results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                # r.keypoints.xy expected to be iterable over people, each a tensor
                for person_kpts in getattr(r.keypoints, "xy", []) or getattr(r.keypoints, "xyxy", []) or getattr(r.keypoints, "xy", []):
                    # handle if person_kpts is a torch tensor
                    try:
                        kp_tensor = person_kpts
                        # some ultralytics versions use .xy or .xyxy; handle common cases:
                        # if kp_tensor is a torch tensor, convert to numpy
                        keypoints_np = None
                        if hasattr(kp_tensor, "cpu"):
                            keypoints_np = kp_tensor.cpu().numpy()
                        elif isinstance(kp_tensor, np.ndarray):
                            keypoints_np = kp_tensor
                        else:
                            # try to coerce
                            keypoints_np = np.array(kp_tensor)
                    except Exception:
                        continue

                    if keypoints_np is None or len(keypoints_np) == 0:
                        continue

                    # Ensure shape (N,2) or (N,3)
                    if keypoints_np.ndim == 1:
                        # weird shape; skip
                        continue
                    if keypoints_np.shape[1] < 2:
                        continue

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

                    # Extract x, y (handle if kp has confidence column)
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

                    pose_label = "facing_camera"
                    if offset_x > H_THRESH:
                        pose_label = "looking_right"
                    elif offset_x < -H_THRESH:
                        pose_label = "looking_left"
                    elif offset_y > DOWN_THRESH:
                        pose_label = "looking_down"
                    elif offset_y < UP_THRESH:
                        pose_label = "looking_up"
                    else:
                        pose_label = "facing_camera"

                    # Draw label near nose
                    x, y = int(nx), int(ny) - 10
                    cv2.putText(frame, pose_label, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Log pose result
                    log_msg = f"[{cam_name}] Pose label: {pose_label}"
                    print(log_msg)
                    logging.info(log_msg)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# === Live Log Streaming via SSE ===
def follow(logfile):
    logfile.seek(0, 2)  # Go to end
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"

# === Flask routes ===
@app.route('/')
def index():
    # Buttons switch the <img src> to the chosen camera
    template = """
    <html>
    <head>
        <title>Argus - Multi-Camera Cheating Detection</title>
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
            function switchCam(camName) {
                document.getElementById("videoStream").src = "/video/" + camName;
            }

            var evtSource = new EventSource("/log_stream");
            var logDiv = document.getElementById("log");
            evtSource.onmessage = function(e) {
                logDiv.innerHTML += e.data + "<br>";
                logDiv.scrollTop = logDiv.scrollHeight;
            };
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
    logfile = open(log_file, "r")
    return Response(follow(logfile), mimetype="text/event-stream")

if __name__ == "__main__":
    # threaded=True to allow SSE + video streams to work concurrently
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

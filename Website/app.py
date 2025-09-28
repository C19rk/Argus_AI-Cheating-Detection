from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import logging
import time
import threading

app = Flask(__name__)

# === Logging Setup ===
log_file = "detections.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# === Load YOLO model ===
yolo_model_path = "../App/runs/aidetection4/weights/best.pt"
yolo = YOLO(yolo_model_path)

# === Camera RTSP URLs ===
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


def generate_frames(cam_name):
    cap = caps[cam_name]
    while True:
        success, frame = cap.read()
        if not success:
            continue

        results = yolo.predict(frame, imgsz=640, conf=0.25)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                label = yolo.names[cls]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Draw YOLO box
                color = (255, 0, 0) if label == "person" else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Log detection (always)
                log_msg = f"[{cam_name}] Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]"
                print(log_msg)
                logging.info(log_msg)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# === Live Log Streaming via SSE ===
def follow(logfile):
    logfile.seek(0, 2)  # Go to end
    while True:
        line = logfile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield f"data: {line}\n\n"


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
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

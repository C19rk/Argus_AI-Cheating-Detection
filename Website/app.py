from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import logging
import time

app = Flask(__name__)

# === Logging Setup ===
log_file = "detections.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# === Load YOLO model ===
yolo_model_path = "../App/runs/aidetection2/weights/best.pt"
yolo = YOLO(yolo_model_path)

# === Camera ===
rtsp_url = "rtsp://administrator:admin123@192.168.100.203:554/stream2"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def generate_frames():
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

                # Log detection
                log_msg = f"Detected {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]"
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
    # Flexbox layout: video on left, log on right
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video">
                <h2>Live Video</h2>
                <img src="/video">
            </div>
            <div class="log">
                <h2>Live Detection Log</h2>
                <div id="log"></div>
            </div>
        </div>
        <script>
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


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/log_stream')
def log_stream():
    logfile = open(log_file, "r")
    return Response(follow(logfile), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

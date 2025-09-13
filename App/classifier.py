from ultralytics import YOLO
import cv2
import torch
import numpy as np
from collections import deque

# ==================== CONFIG ====================
FRAME_HISTORY = 10
LOOKING_AWAY_THRESHOLD = 120
HEAD_DOWN_THRESHOLD = 30
MOUTH_OPEN_THRESHOLD = 12
HAND_FACE_DIST = 90

history_queue = deque(maxlen=FRAME_HISTORY)

# COCO skeleton (17 keypoints)
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 6], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

# ==================== HELPERS ====================
def pad_kpt(kp):
    kp = np.array(kp)
    if kp.size == 2:
        return np.array([kp[0], kp[1], 1.0])
    elif kp.size == 3:
        return kp
    else:
        return np.array([0, 0, 0])

person_history = {}
mouth_history = {}

FRAME_HISTORY = 8
MOUTH_HISTORY = 5
MOUTH_MOVE_THRESHOLD = 3
MIN_CONF = 0.3
LOOKING_AWAY_RATIO = 0.25
HEAD_DOWN_RATIO = 0.2
HAND_FACE_RATIO = 0.25

NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_MOUTH, RIGHT_MOUTH = 13, 14

def safe_kpt(kpt):
    kpt = np.array(kpt)
    if kpt.size == 2:
        return np.array([kpt[0], kpt[1], 1.0])
    elif kpt.size == 3:
        return kpt
    else:
        return np.array([0, 0, 0])

def classify_pose(keypoints, pid=0):
    keypoints = [safe_kpt(k) for k in keypoints]

    nose = keypoints[NOSE]
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    left_wrist = keypoints[LEFT_WRIST]
    right_wrist = keypoints[RIGHT_WRIST]
    left_mouth = keypoints[LEFT_MOUTH]
    right_mouth = keypoints[RIGHT_MOUTH]

    if pid not in person_history:
        person_history[pid] = deque(maxlen=FRAME_HISTORY)
    if pid not in mouth_history:
        mouth_history[pid] = deque(maxlen=MOUTH_HISTORY)

    if left_shoulder[2] < MIN_CONF or right_shoulder[2] < MIN_CONF:
        shoulder_dist = 1.0
        shoulder_center_x = nose[0]
        shoulder_center_y = nose[1]
    else:
        shoulder_dist = np.linalg.norm(left_shoulder[:2]-right_shoulder[:2]) + 1e-6
        shoulder_center_x = (left_shoulder[0]+right_shoulder[0])/2
        shoulder_center_y = (left_shoulder[1]+right_shoulder[1])/2

    new_label = "normal"

    if nose[2] >= MIN_CONF:
        nose_offset = abs(nose[0]-shoulder_center_x)/shoulder_dist
        if nose_offset > LOOKING_AWAY_RATIO:
            new_label = "looking_away"

    nose_y_rel = (nose[1]-shoulder_center_y)/shoulder_dist
    if nose[2]>=MIN_CONF and nose_y_rel > HEAD_DOWN_RATIO:
        new_label = "head_down"

    left_hand_dist = np.linalg.norm(left_wrist[:2]-nose[:2])/shoulder_dist if left_wrist[2]>=MIN_CONF else 1
    right_hand_dist = np.linalg.norm(right_wrist[:2]-nose[:2])/shoulder_dist if right_wrist[2]>=MIN_CONF else 1
    if left_hand_dist < HAND_FACE_RATIO or right_hand_dist < HAND_FACE_RATIO:
        new_label = "phone"

    if left_mouth[2]>=MIN_CONF and right_mouth[2]>=MIN_CONF:
        mouth_y = (left_mouth[1]+right_mouth[1])/2
        mh = mouth_history[pid]
        if mh:
            delta = abs(mouth_y - mh[-1])
            if delta > MOUTH_MOVE_THRESHOLD and new_label == "normal":
                new_label = "talking"
        mh.append(mouth_y)

    ph = person_history[pid]
    if not ph or ph[-1] != new_label:
        ph.append(new_label)

    label = max(set(ph), key=ph.count)
    return label

def draw_skeleton(frame, kpts):
    kpts = [pad_kpt(k) for k in kpts]
    for start, end in SKELETON:
        if start < len(kpts) and end < len(kpts):
            if kpts[start][2] > 0.3 and kpts[end][2] > 0.3:
                x1, y1 = int(kpts[start][0]), int(kpts[start][1])
                x2, y2 = int(kpts[end][0]), int(kpts[end][1])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for x, y, v in kpts:
        if v > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

# ==================== MAIN ====================
def main():
    model = YOLO("yolo11n-pose.pt")
    device = 0 if torch.cuda.is_available() else 'cpu'
    print("Using GPU" if device==0 else "GPU not found, using CPU")

    results = model.track(
        source=0,
        show=False,
        stream=True,
        conf=0.5,
        device=device,
        project="App/runs",
        name="pose"
    )

    for r in results:
        frame = r.orig_img
        keypoints = r.keypoints.xy if r.keypoints is not None else None
        boxes = r.boxes.xyxy if r.boxes is not None else None

        if keypoints is not None:
            for i, person_kpts in enumerate(keypoints):
                person_np = person_kpts.cpu().numpy()
                label = classify_pose(person_np, i)
                draw_skeleton(frame, person_np)

                nose = person_np[0]
                cv2.putText(frame, f"ID{i}: {label}", (int(nose[0]), int(nose[1]-15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Argus Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

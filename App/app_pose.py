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
    return np.array([0, 0, 0])

def classify_pose(keypoints):
    if keypoints.shape[0] < 1:
        return "normal"

    keypoints = np.array([pad_kpt(k) for k in keypoints])

    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_mouth = keypoints[13]
    right_mouth = keypoints[14]

    # HEAD & EYE
    eyes_visible = (left_eye[2] > 0.3) and (right_eye[2] > 0.3)
    nose_visible = nose[2] > 0.3
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    nose_offset = abs(nose[0] - shoulder_center_x)

    label = "normal"
    if not eyes_visible or not nose_visible or nose_offset > LOOKING_AWAY_THRESHOLD:
        label = "looking_away"
    elif nose[1] > max(left_shoulder[1], right_shoulder[1]) + HEAD_DOWN_THRESHOLD:
        label = "head_down"

    # TALKING
    mouth_open = False
    if left_mouth[2] > 0.3 and right_mouth[2] > 0.3:
        mouth_open = abs(left_mouth[1] - right_mouth[1]) > MOUTH_OPEN_THRESHOLD
    if mouth_open and label == "normal":
        label = "talking"

    # PHONE
    left_hand_near_face = np.linalg.norm(left_wrist[:2] - nose[:2]) < HAND_FACE_DIST
    right_hand_near_face = np.linalg.norm(right_wrist[:2] - nose[:2]) < HAND_FACE_DIST
    if left_hand_near_face or right_hand_near_face:
        label = "phone"

    history_queue.append(label)
    if history_queue:
        return max(set(history_queue), key=history_queue.count)
    return "normal"

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
                label = classify_pose(person_np)
                draw_skeleton(frame, person_np)

                if boxes is not None:
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID{i}: {label}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Argus Pose Multi-Person", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from ultralytics import YOLO

def main():
    # Load pretrained model
    model = YOLO("yolo11n-pose.pt")

    # Run validation on COCO keypoints (just as demo, no local dataset needed)
    model.val(
        data="coco8-pose.yaml",  # built-in mini COCO dataset
        imgsz=640,
        device="cpu"             # or 0 if you have GPU
    )

if __name__ == "__main__":
    main()

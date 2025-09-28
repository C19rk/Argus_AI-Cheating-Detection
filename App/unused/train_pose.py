from ultralytics import YOLO

def main():
    # Load YOLOv8n-pose pretrained weights
    model = YOLO("yolov8n-pose.pt")

    # Train on your custom pose dataset
    model.train(
        data="data_pose.yaml",   # pose dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",            # change to "0" if you actually have GPU
        name="aidestimation"     # run folder name
    )

if __name__ == "__main__":
    main()

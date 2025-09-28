from ultralytics import YOLO

def main():
    # Load trained model (best.pt from training)
    model = YOLO("runs/pose/train/weights/best.pt")

    # Validate
    model.val(
        data="mypose.yaml",
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    main()

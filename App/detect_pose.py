from ultralytics import YOLO
import cv2

def main():
    # Load pretrained model
    model = YOLO("yolo11n-pose.pt")

    # Run inference on an image
    results = model("test.jpg", imgsz=640, conf=0.5, save=True)

    # Show result
    for r in results:
        im = r.plot()
        cv2.imshow("Pose Detection", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

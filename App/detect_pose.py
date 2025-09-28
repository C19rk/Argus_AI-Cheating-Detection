from ultralytics import YOLO
import cv2

def main():
    # Load trained pose model
    model = YOLO("runs/pose/train/weights/best.pt")

    # Run inference on an image
    results = model("test.jpg", imgsz=640, conf=0.5, save=True)

    # Display predictions
    for r in results:
        im = r.plot()  # numpy array with boxes + keypoints drawn
        cv2.imshow("Pose", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

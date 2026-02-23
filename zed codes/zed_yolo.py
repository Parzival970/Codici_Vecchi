import pyzed.sl as sl
import cv2
import torch
import numpy as np
from ultralytics import YOLO

model_path = "/home/omni/yolo11n.pt"
model = YOLO(model_path)

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

image_zed = sl.Mat()

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            # Convert image to NumPy array correctly
            frame = image_zed.get_data()
            frame = np.copy(frame)  # Ensure it's a proper NumPy array

            # Ensure frame is a valid NumPy array
            if not isinstance(frame, np.ndarray):
                print("Failed to convert ZED image to NumPy array")
                continue

            # Ensure frame has 3 color channels (remove alpha if present)
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # YOLO Inference
            results = model(frame)

            # Draw detected objects1q
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = box.conf[0].item()  # Confidence score
                    cls = int(box.cls[0].item())  # Class ID

                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show frame with detections
            cv2.imshow("ZED + YOLOv11 Object Detection", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
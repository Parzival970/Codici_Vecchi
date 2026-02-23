import pyzed.sl as sl
import cv2
import torch
import numpy as np
from ultralytics import YOLO

model_path = "/home/omni/esempi_py/yolo11m-seg.pt"
# Load YOLOv11 model
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
                masks = result.masks
                if masks is not None:
                    mask_data = masks.data  # This should be a tensor of shape [n, h, w]
                    for i in range(mask_data.shape[0]):
                        mask = mask_data[i].cpu().numpy()  # Convert tensor to NumPy
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        binary_mask = (mask_resized * 255).astype(np.uint8)
                        colored_mask = cv2.merge([binary_mask] * 3)
                        if colored_mask.shape == frame.shape:
                            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            # Show frame with detections
            cv2.imshow("ZED + YOLOv11 Object Detection", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
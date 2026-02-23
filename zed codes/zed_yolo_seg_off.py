# Import required libraries
import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d
import os

# Load pre-trained YOLO model
model_path = "/home/omni/esempi_py/yolo11x-seg.pt"
model = YOLO(model_path)

# Set up ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
zed.open(init_params)

# Initialize containers
runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

# Prepare output folder
os.makedirs("recorded_frames", exist_ok=True)
video_path = "recorded_frames/apple_segmentation.avi"
video_writer = None
frame_count = 0

# Main loop to record a few seconds
while frame_count < 150:  # ~10 seconds at 15 FPS
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        if video_writer is None:
            height, width = img_np.shape[:2]
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 15, (width, height))

        results = model(img_np, verbose=False)[0]
        masks = results.masks
        names = model.names
        overlay = img_np.copy()

        if masks is not None:
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                if class_name != "apple":
                    continue

                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5
                print(mask_resized.shape, pc_np.shape)
                overlay[mask_binary] = [0, 255, 0]

                np.save(f"recorded_frames/pc_{frame_count:04d}.npy", pc_np)
                np.save(f"recorded_frames/mask_{frame_count:04d}.npy", mask_binary)
                break  # Save only first apple for now

        video_writer.write(overlay)
        cv2.imshow("Recording", overlay)
        frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_writer.release()
zed.close()
cv2.destroyAllWindows()
print("Recording complete. Data saved to 'recorded_frames/'.")

import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO

model_path = "/home/omni/esempi_py/yolo11m-seg.pt"
# Load YOLOv11 model
model = YOLO(model_path)

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
zed.open(init_params)

runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)  # Convert to BGR
        pc_np = point_cloud.get_data()

        # Inference with YOLO
        results = model(img_np, verbose=False)[0]
        masks = results.masks

        if masks is not None:
            for i, mask in enumerate(masks.data):
                # Resize the mask to match point cloud dimensions
                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5  # Threshold the mask to binary

                # Apply the mask to the point cloud
                segmented_points = pc_np[mask_binary]

                # Filter out invalid points
                valid_points = segmented_points[~np.isnan(segmented_points).any(axis=1)]

                print(f"Object {i}: {valid_points.shape[0]} valid points")

                # Optional: save as.ply
                # save_ply(valid_points, f"object_{i}.ply")
        
        # Display the image with masks
        overlay = img_np.copy()
        overlay[mask_binary] = [0, 255, 0]  # green overlay
        cv2.imshow(f"Object {i}", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
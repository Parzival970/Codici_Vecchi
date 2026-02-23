import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO

model_path = "/home/omni/esempi_py/yolo11x-seg.pt"
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
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        results = model(img_np, verbose=False)[0]
        
        
        masks = results.masks
        names = model.names
        target_class = "apple"

        # Reset overlay image
        overlay = img_np.copy()
        found_target = False

        if masks is not None:
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                print(class_name)
                if target_class not in class_name:
                    continue

                found_target = True
                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.1
                segmented_points = pc_np[mask_binary]
                valid_points = segmented_points[~np.isnan(segmented_points).any(axis=1)]

                print(f"Object {i} ({class_name}): {valid_points.shape[0]} valid points")

                # Draw overlay
                overlay[mask_binary] = [0, 255, 0]
                break  # Only process the first detected apple

        if found_target:
            cv2.imshow("Target Object Mask", overlay)
        else:
            cv2.imshow("Target Object Mask", img_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()

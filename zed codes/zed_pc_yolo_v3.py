import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO

def save_ply(points, filename):
    """Save point cloud to PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")  # Required line
        for p in points:
            x, y, z, rgba = p
            r = int(rgba) & 255
            g = (int(rgba) >> 8) & 255
            b = (int(rgba) >> 16) & 255
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

model_path = "/home/omni/esempi_py/yolo11m-seg.pt"
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

                if class_name != target_class:
                    continue

                found_target = True
                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5
                segmented_points = pc_np[mask_binary]
                valid_points = segmented_points[~np.isnan(segmented_points).any(axis=1)]

                print(f"Object {i} ({class_name}): {valid_points.shape[0]} valid points")

                # Save to PLY
                save_ply(valid_points, f"apple_segmented_{i}.ply")
                print(f"Saved to apple_segmented_{i}.ply")

                # Draw overlay
                overlay[mask_binary] = [0, 255, 0]
                break  # Only process the first detected apple

        # Display
        cv2.imshow("Target Object Mask", overlay if found_target else img_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d

# Load YOLOv8 model
model = YOLO("/home/omni/esempi_py/yolo11x-seg.pt")

# Initialize ZED
zed = sl.Camera()
init = sl.InitParameters(
    depth_mode=sl.DEPTH_MODE.ULTRA,
    coordinate_units=sl.UNIT.MILLIMETER
)
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED failed to open.")
    exit()

runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

print("Press 'q' to quit")

while True:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        results = model(img_bgr, verbose=False)[0]
        masks = results.masks
        names = model.names

        if masks is not None:
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                if class_name != "apple":
                    continue

                print(f"Apple {i} detected — extracting 3D points")

                # Resize mask to match point cloud
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5

                # Apply mask
                masked_pc = pc_np[mask_binary]
                valid_points = masked_pc[~np.isnan(masked_pc).any(axis=1)]

                if valid_points.shape[0] == 0:
                    print("No valid 3D points found.")
                    continue

                # Convert to Open3D format
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3] / 1000.0)  # mm → m
                pcd.paint_uniform_color([1.0, 0.2, 0.2])

                print(f"Showing {valid_points.shape[0]} apple points in 3D...")

                # Show once
                o3d.visualization.draw_geometries([pcd], window_name="Apple Point Cloud")
                break  # show only first detected apple

        # Show segmentation overlay
        overlay = img_bgr.copy()
        if masks is not None:
            for mask, cls in zip(masks.data, results.boxes.cls):
                if names[int(cls)] != "apple":
                    continue
                m = cv2.resize(mask.cpu().numpy(), (img_bgr.shape[1], img_bgr.shape[0]))
                overlay[m > 0.5] = [0, 255, 0]
        cv2.imshow("Apple Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

zed.close()
cv2.destroyAllWindows()

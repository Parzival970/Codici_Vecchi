import pyzed.sl as sl
import numpy as np
import cv2
from ultralytics import YOLO
import open3d as o3d

# Load YOLOv8 segmentation model
model = YOLO("/home/omni/esempi_py/yolo11x-seg.pt")  # consider switching to yolov8n-seg.pt

# Init ZED
zed = sl.Camera()
init_params = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA, coordinate_units=sl.UNIT.MILLIMETER)
zed.open(init_params)

runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

# Open3D visualizer setup
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Apple Point Cloud (Live)", width=800, height=600)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

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
        all_valid_points = []

        # Draw overlay
        overlay = img_bgr.copy()

        if masks is not None:
            for mask, cls in zip(masks.data, results.boxes.cls):
                class_name = names[int(cls)]
                if class_name != "apple":
                    continue

                # Resize mask
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5
                overlay[mask_binary] = [0, 255, 0]

                # Extract valid 3D points
                masked_pc = pc_np[mask_binary]
                valid_points = masked_pc[~np.isnan(masked_pc).any(axis=1)]
                if valid_points.shape[0] > 0:
                    all_valid_points.append(valid_points[:, :3])

        # Update Open3D point cloud
        if all_valid_points:
            combined_points = np.vstack(all_valid_points) / 1000.0  # mm to m
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            pcd.paint_uniform_color([1.0, 0.2, 0.2])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if first_frame:
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                ctr.set_zoom(0.5)
                first_frame = False

        cv2.imshow("Apple Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vis.destroy_window()
zed.close()
cv2.destroyAllWindows()

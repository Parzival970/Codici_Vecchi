# Import required libraries
import pyzed.sl as sl          # Zed camera SDK
import numpy as np             # Numerical operations
import cv2                     # OpenCV for image processing
from ultralytics import YOLO   # YOLO model for object detection
import open3d as o3d           # Open3D for 3D point cloud visualization
import matplotlib.pyplot as plt

# Load pre-trained YOLO model for object segmentation
model_path = "/home/omni/esempi_py/yolo11x-seg.pt"
model = YOLO(model_path)

# Set up ZED stereo camera with specific settings
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution to 720p
init_params.depth_mode = sl.DEPTH_MODE.ULTRA         # High quality depth mapping
init_params.coordinate_units = sl.UNIT.MILLIMETER    # Use mm for measurements
zed.open(init_params)

# Initialize variables for image
runtime = sl.RuntimeParameters()
image = sl.Mat()                   # Image container for left camera view
point_cloud = sl.Mat()             # Point cloud container for depth data

def estimate_curvature_from_masked_pointcloud(pc_np, mask_binary, voxel_size=1.0):
    points = pc_np[mask_binary][:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # Mesh reconstruction using Poisson
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for visibility

    # Estimate curvature from point cloud (not the mesh for speed)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    r1_list, r2_list, centers = [], [], []

    for i, point in enumerate(pcd.points):
        _, idx, _ = pcd_tree.search_knn_vector_3d(point, 30)
        neighbors = np.asarray(pcd.points)[idx, :]
        if neighbors.shape[0] < 3:
            continue
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        cov = neighbors_centered.T @ neighbors_centered
        evals, _ = np.linalg.eigh(cov)
        evals = np.sort(evals)
        k1, k2 = evals[0], evals[1]
        if k1 + k2 == 0:
            continue
        r1_list.append(1.0 / np.sqrt(k1 + 1e-8))
        r2_list.append(1.0 / np.sqrt(k2 + 1e-8))
        centers.append(point)

    if not r1_list or not r2_list:
        return None

    # Grasp marker on flattest region (largest radius)
    flat_idx = np.argmax(r1_list)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    marker.paint_uniform_color([0, 1, 0])
    marker.translate(centers[flat_idx])

    o3d.visualization.draw_geometries([mesh, marker], mesh_show_back_face=True)

    return {
        "mean_radius_1": np.median(r1_list),
        "mean_radius_2": np.median(r2_list),
        "mean_curvature": np.median([1.0 / r for r in r1_list]) / 2 + np.median([1.0 / r for r in r2_list]) / 2,
        "gaussian_curvature": np.median([1.0 / r for r in r1_list]) * np.median([1.0 / r for r in r2_list])
    }

# Main processing loop
while True:
    # Capture a new frame from the ZED camera
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve both 2D image and 3D point cloud from camera
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        results = model(img_np, verbose=False)[0]
        masks = results.masks
        names = model.names

        overlay = img_np.copy()

        if masks is not None:
            apple_index = 0
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                if class_name != "apple":
                    continue

                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5

                overlay[mask_binary] = [0, 255, 0]
                print(f"Detected apple {apple_index}: {np.sum(mask_binary)} pixels")

                curvature = estimate_curvature_from_masked_pointcloud(pc_np, mask_binary)
                if curvature:
                    print(f" -> Curvature for apple {apple_index}: ")
                    print(f"    Radius 1 (R1): {curvature['mean_radius_1']:.2f} mm")
                    print(f"    Radius 2 (R2): {curvature['mean_radius_2']:.2f} mm")
                    print(f"    Mean Curvature (H): {curvature['mean_curvature']:.5f}")
                    print(f"    Gaussian Curvature (K): {curvature['gaussian_curvature']:.5f}")

                apple_index += 1

        cv2.imshow("Apple Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()

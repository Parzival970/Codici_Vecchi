import sys
import os
import json
import time
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "/home/omni/esempi_py/yolo11m-seg.pt"  # deve essere YOLOv8-seg
OUTPUT_VIDEO = "ball_output.avi"
BBOX_JSON = "ball_bbox_data.json"
POINTCLOUD_DIR = "ball_pointclouds"
DURATION_SEC = 2
FPS = 15

# Create output dir
os.makedirs(POINTCLOUD_DIR, exist_ok=True)

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Init ZED ===
init = sl.InitParameters(
    depth_mode=sl.DEPTH_MODE.NEURAL,
    coordinate_units=sl.UNIT.METER,
    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
)
zed = sl.Camera()
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED failed to open.")
    exit()

# Get calibration parameters
calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

# Intrinsics of the left camera
fx = calibration_params.left_cam.fx
fy = calibration_params.left_cam.fy
cx = calibration_params.left_cam.cx
cy = calibration_params.left_cam.cy

print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

# === Setup ZED parameters ===
res = sl.Resolution(1280, 720)
runtime = sl.RuntimeParameters()
zed_image = sl.Mat()
point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

# === Setup VideoWriter ===
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (res.width, res.height))

# === Bounding box data ===
bbox_data = {}

print("Recording started...")

frame_idx = 0
start_time = time.time()

while frame_idx < DURATION_SEC * FPS:
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # === Capture image and point cloud ===
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)

        img_np = zed_image.get_data()
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        results = model(img_rgb, verbose=False)[0]

        # === Conta palline ===
        num_ball = 0
        if results.masks is not None:
            for cls_id in results.boxes.cls.cpu().numpy().astype(int):
                if model.names[cls_id] == "sports ball":
                    num_ball += 1

        print(f"[Frame {frame_idx}] 🎯 Palline trovate: {num_ball}")

        frame_name = f"frame_{frame_idx:04d}"
        bbox_data[frame_name] = []

        # === Maschera segmentata ===
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)  # (N, H_seg, W_seg)

            # Ridimensiona maschere alla risoluzione ZED (720x1280)
            masks_resized = []
            for m in masks:
                m_resized = cv2.resize(m, (res.width, res.height), interpolation=cv2.INTER_NEAREST)
                masks_resized.append(m_resized)
            masks = np.array(masks_resized)

            for i, (cls_id, mask) in enumerate(zip(results.boxes.cls, masks)):
                class_name = model.names[int(cls_id)]
                print (class_name)
                if class_name != "sports ball":
                    continue

                mask_bool = mask > 0
                xyzrgba = point_cloud.get_data()  # shape (H, W, 4)
                crop = xyzrgba[mask_bool, :3]

                valid = np.isfinite(crop).all(axis=1)
                crop = crop[valid]
                crop = crop[np.linalg.norm(crop, axis=1) < 5.0]  # max 5 metri

                bbox_data[frame_name].append({
                    "class": class_name,
                    "mask_index": i,
                    "num_points": len(crop)
                })

                if crop.shape[0] > 0:
                    ball_ply = os.path.join(POINTCLOUD_DIR, f"ball_{frame_name}.ply")
                    with open(ball_ply, 'w') as f:
                        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(crop)))
                        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                        for p in crop:
                            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # === Save frame to video ===
        video_out.write(img_rgb)

        # === Save full point cloud ===
        ply_filename = os.path.join(POINTCLOUD_DIR, f"pointcloud_{frame_idx:04d}.ply")
        point_cloud.write(ply_filename)

        frame_idx += 1

# === Save bbox/mask data ===
with open(BBOX_JSON, 'w') as f:
    json.dump(bbox_data, f, indent=4)

# === Cleanup ===
video_out.release()
zed.close()

print(f"\n✅ Acquisizione completata: {frame_idx} frame salvati.")
print(f"🎞️ Video: {OUTPUT_VIDEO}")
print(f"📁 Point cloud: {POINTCLOUD_DIR}")
print(f"📄 Dati maschera: {BBOX_JSON}")
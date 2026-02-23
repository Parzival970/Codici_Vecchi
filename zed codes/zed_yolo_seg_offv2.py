import sys
import os
import json
import time
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "/home/omni/esempi_py/yolo11x-seg.pt"
OUTPUT_VIDEO = "ball_output.avi"
BBOX_JSON = "ball_bbox_data.json"
POINTCLOUD_DIR = "ball_pointclouds"
DURATION_SEC = 2
FPS = 30

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

        frame_name = f"frame_{frame_idx:04d}"
        bbox_data[frame_name] = []

        # === Draw bboxes ===
        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                class_name = model.names[int(cls)]
                if class_name != "sports ball":
                    continue
                x1, y1, x2, y2 = [int(b.item()) for b in box]
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bbox_data[frame_name].append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2]
                })

        # === Save frame to video ===
        video_out.write(img_rgb)

        # === Save point cloud ===
        ply_filename = os.path.join(POINTCLOUD_DIR, f"pointcloud_{frame_idx:04d}.ply")
        point_cloud.write(ply_filename)

        frame_idx += 1

# === Save bbox data ===
with open(BBOX_JSON, 'w') as f:
    json.dump(bbox_data, f, indent=4)

# === Cleanup ===
video_out.release()
zed.close()

print(f"\n✅ Acquisizione completata: {frame_idx} frame salvati.")
print(f"🎞️ Video: {OUTPUT_VIDEO}")
print(f"📁 Point cloud: {POINTCLOUD_DIR}")
print(f"📄 Bounding box: {BBOX_JSON}")

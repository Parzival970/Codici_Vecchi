import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 segmentation model
model = YOLO("/home/omni/esempi_py/yolo11x-seg.pt")

# Initialize ZED camera
init = sl.InitParameters(
    depth_mode=sl.DEPTH_MODE.QUALITY,
    coordinate_units=sl.UNIT.METER,
    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
)

zed = sl.Camera()
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED failed to open.")
    exit()

res = sl.Resolution(720, 404)
runtime = sl.RuntimeParameters()

# Init viewer
viewer = gl.GLViewer()
viewer.init(1, sys.argv, res)

# Allocate ZED image and point cloud containers
zed_image = sl.Mat()
point_cloud_full = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
point_cloud_filtered = sl.Mat()
point_cloud_full.copy_to(point_cloud_filtered)

print("Press 'ESC' to exit.")

while viewer.is_available():
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Get RGB image and full point cloud
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud_full, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)

        # Convert image to numpy for YOLO
        img_np = zed_image.get_data()
        img_rgb = img_np[:, :, :3][:, :, ::-1].copy()  # Convert BGRA to RGB

        results = model(img_rgb, verbose=False)[0]
        masks = results.masks
        names = model.names

        # Clear filtered cloud
        # point_cloud_filtered.set_to([0.0, 0.0, 0.0, 0.0])

        if masks is not None:
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                if class_name != "apple":
                    continue

                print(f"Apple {i} detected — filtering point cloud.")

                # Resize mask to match resolution
                # mask_resized = mask.cpu().numpy()
                # mask_resized = np.array(mask_resized * 255, dtype=np.uint8)
                # mask_resized = cv2.resize(mask_resized, (res.width, res.height))
                # mask_binary = mask_resized > 127
        
                # Mask point cloud
                # pc_np = point_cloud_full.get_data()
                #pc_filtered_np = np.zeros_like(pc_np)
                #pc_filtered_np[mask_binary] = pc_np[mask_binary]
                # point_cloud_filtered.get_data()[:] = pc_filtered_np
                mask_np = mask.cpu().numpy()
        
                # Gestisci le dimensioni
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()
                
                # Normalizza i valori
                if mask_np.max() <= 1.0:
                    mask_np = mask_np * 255
                    
                mask_uint8 = mask_np.astype(np.uint8)
                
                # Ridimensiona con interpolazione appropriata
                mask_resized = cv2.resize(mask_uint8, (res.width, res.height), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Crea maschera binaria
                mask_binary = mask_resized > 127
                
                # Verifica compatibilità dimensioni
                pc_np = point_cloud_full.get_data()
                
                if pc_np.shape[:2] != mask_binary.shape:
                    print(f"Warning: Shape mismatch - PC: {pc_np.shape}, Mask: {mask_binary.shape}")
                    
                # Applica la maschera
                pc_filtered_np = np.zeros_like(pc_np)
                pc_filtered_np[mask_binary] = pc_np[mask_binary]
                
                # Aggiorna la point cloud
                point_cloud_filtered.get_data()[:] = pc_filtered_np


        # Update viewer with filtered cloud
        viewer.updateData(point_cloud_filtered)

        # Optionally save
        if viewer.save_data:
            ply = sl.Mat()
            zed.retrieve_measure(ply, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
            if ply.write("filtered_apples.ply") == sl.ERROR_CODE.SUCCESS:
                print("Saved filtered point cloud as PLY.")
            else:
                print("Failed to save point cloud.")
            viewer.save_data = False

# Clean up
viewer.exit()
zed.close()
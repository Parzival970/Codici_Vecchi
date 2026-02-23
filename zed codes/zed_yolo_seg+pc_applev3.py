import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("/home/omni/esempi_py/yolo11x-seg.pt")  # può anche essere detection-only

# Init ZED
init = sl.InitParameters(
    depth_mode=sl.DEPTH_MODE.NEURAL,
    coordinate_units=sl.UNIT.MILLIMETER,
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

# Allocate image and cloud
zed_image = sl.Mat()
point_cloud_full = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
point_cloud_filtered = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

print("Press 'ESC' to exit.")

while True: # viewer.is_available():
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Get image and point cloud
        zed.retrieve_image(zed_image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud_full, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)

        img_np = zed_image.get_data()
        #img_rgb = img_np[:, :, :3][:, :, ::-1].copy()  # BGRA to RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        #frame_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0        
        results = model(img_rgb, verbose=False)[0]
        #results = model(frame_tensor.unsqueeze(0), verbose=False)[0]
        names = model.names
        boxes = results.boxes

        pc_np = point_cloud_full.get_data()
        pc_filtered_np = np.zeros_like(pc_np)  # inizializza qui, una volta sola
        
        img_display = img_rgb.copy()

        if boxes is not None:

            apple_idex  = 0

            for i, (box, cls_id) in enumerate(zip(boxes.xyxy, boxes.cls)):
                class_name = names[int(cls_id)]
                if class_name != "apple":
                    continue
                                
                x1, y1, x2, y2 = [int(v.item()) for v in box[:4]]

                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_display, f"Apple {apple_idex}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Apple {apple_idex}: bounding box = ({x1}, {y1}) to ({x2}, {y2})")
                apple_idex += 1

                patch = pc_np[y1:y2, x1:x2] 

                if patch.shape[2] == 3:
                    confidence_channel = np.full(patch.shape[:2] + (1,), -2e+38, dtype=np.float32)
                    patch = np.concatenate([patch, confidence_channel], axis=2)
                
                # Filtro 3D: solo punti validi
                xyz = patch[:, :, :3]
                confidence = patch[:, :, 3]
                valid_mask = (
                    np.isfinite(xyz).all(axis=2) &
                    (xyz[:,:, 2] > 0.1) & (xyz[:,:,2] < 2.5) &
                    (confidence > -1e+20)
                )
                print("Valid points in patch:", np.count_nonzero(valid_mask))
                patch_filtered = np.zeros_like(patch)
                patch_filtered[valid_mask] = patch[valid_mask]
                pc_filtered_np[y1:y2, x1:x2] = patch_filtered
                print(f"Filtered patch shape: {patch_filtered.shape}")
        
            point_cloud_filtered.get_data()[:] = pc_filtered_np

        viewer.updateData(point_cloud_filtered)

        cv2.imshow("ZED Image", img_display)
        key = cv2.waitKey(1)
        if key == 27:
            print("Exiting...")
            break

        if viewer.save_data:
            if point_cloud_filtered.write("filtered_apples_bbox.ply") == sl.ERROR_CODE.SUCCESS:
                print("Saved PLY with bounding boxes.")
            else:
                print("Failed to save PLY.")
            viewer.save_data = False

# Cleanup
viewer.exit()
zed.close()
cv2.destroyAllWindows()
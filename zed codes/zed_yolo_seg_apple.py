# Import required libraries
import pyzed.sl as sl          # Zed camera SDK
import numpy as np             # Numerical operations
import cv2                     # OpenCV for image processing
from ultralytics import YOLO   # YOLO model for object detection

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

# Main processing loop
while True:
    # Capture a new frame from the ZED camera
    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        # Retrieve both 2D image and 3D point cloud from camera
        zed.retrieve_image(image, sl.VIEW.LEFT)                 # Get left camera image
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)   # Get point cloud data (XYZ + RGBA)

        # Convert captured image to numpy array and change color format
        img_np = image.get_data()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()

        # Run YOLO model on the captured image
        results = model(img_np, verbose=False)[0]
        masks = results.masks       # Get segmentation masks from results
        names = model.names         # Class names from the model

        overlay = img_np.copy()     # Create a copy of the image for overlaying masks

        # Process detected objects if any masks are found
        if masks is not None:
            # Initialize index for counting detected apples
            apple_index = 0
            # Iterate through each detected mask and its corresponding class
            for i, (mask, cls) in enumerate(zip(masks.data, results.boxes.cls)):
                class_name = names[int(cls)]
                #print(f"Detected class: {class_name}") # Debugging output for detected class
                # Check if the detected class is "apple"
                if class_name != "apple":
                    continue

                # Resize mask to match point cloud dimensions
                mask_resized = cv2.resize(mask.cpu().numpy(), (pc_np.shape[1], pc_np.shape[0]))
                mask_binary = mask_resized > 0.5

                # Visuall overlay
                overlay[mask_binary] = [0, 255, 0]
                print(f"Detected apple {apple_index}: {np.sum(mask_binary)} pixels")
                apple_index += 1

        cv2.imshow("Apple Segmentation", overlay)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
zed.close()
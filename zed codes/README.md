# ZED + YOLO 3D Vision Toolkit

This project contains a collection of Python scripts for **3D perception using a ZED stereo camera and YOLO models**.

The toolkit allows real-time acquisition, segmentation, and analysis of objects in 3D space using:

* **ZED stereo camera**
* **YOLO (Ultralytics) object detection and segmentation**
* **3D point clouds**
* **Geometric analysis**
* **Dataset recording for offline processing**

---

# Main Features

* Real-time **RGB and depth acquisition**
* **Object detection and segmentation** using YOLO
* **Extraction of object point clouds**
* **Live 3D visualization**
* **Geometric analysis** (curvature, radius estimation)
* **Recording synchronized datasets** (images, masks, point clouds)

---

# Requirements

Install the main Python dependencies:

```bash
pip install ultralytics opencv-python numpy open3d
```

Additional requirements:

* **ZED SDK**
* **pyzed Python API**
* **CUDA** (recommended for YOLO inference)

Official installation guides:

* [https://www.stereolabs.com/docs/](https://www.stereolabs.com/docs/)
* [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

---

# Project Structure

The scripts are organized by functionality.

---

# 1. ZED Camera Tests and Data Acquisition

### `hello_zed.py`

Basic test script to verify that the ZED camera is connected and working correctly.

Prints the camera serial number.

---

### `zed_opencv.py`

Captures and displays:

* RGB images
* depth maps
* point clouds

Allows manual saving of:

* depth images
* point clouds
* stereo images

Useful for **collecting raw datasets**.

---

# 2. YOLO Detection (2D)

### `zed_yolo.py`

Runs YOLO detection on the ZED RGB stream and displays bounding boxes.

Use case:

* quick testing of YOLO models with the ZED camera.

---

### `zed_yolo_fin_v1.py`

### `zed_yolo_fin_v2.py`

Improved versions of the detection pipeline with better handling of:

* image formats
* frame conversion
* inference stability

---

# 3. YOLO Segmentation

### `zed_yolo_seg_apple.py`

Runs **YOLO segmentation** and overlays masks on the RGB image.

Focuses on detecting **apples** as the target object.

Useful for:

* debugging segmentation models
* verifying mask quality.

---

# 4. Segmentation + Point Cloud Extraction

### `zed_pc_yolo.py`

Uses YOLO segmentation masks to extract **3D points belonging to detected objects**.

Displays the number of valid 3D points extracted.

---

### `zed_pc_yolo_v2.py`

Improved version that:

* filters only a **target class** (e.g. apple)
* overlays the mask on the RGB frame.

---

### `zed_pc_yolo_v3.py`

Adds the ability to **save segmented point clouds** as `.ply` files.

Useful for:

* offline 3D analysis
* geometry reconstruction.

---

# 5. Live Filtered Point Cloud Visualization

### `zed_seg_yolo.py`

Filters the ZED point cloud using YOLO segmentation masks and displays the result in a **3D viewer**.

Useful for:

* debugging segmentation in 3D
* verifying object isolation.

---

### `zed_yolo_seg_off.py`

### `zed_yolo_seg_off3.py`

Use **bounding boxes instead of segmentation masks** to filter the point cloud.

Advantages:

* faster processing
* simpler pipeline

Disadvantages:

* includes background points inside the bounding box.

---

# 6. Segmentation + Geometric Analysis

### `zed_yolo_seg+pc_apple.py`

Extracts the segmented point cloud of apples and estimates geometric properties such as:

* curvature
* approximate radius

---

### `zed_yolo_seg+pc_applev2.py`

Extends the previous script by:

* reconstructing a **mesh (Poisson reconstruction)**
* identifying **flat regions suitable for robotic grasping**

Useful for:

* robotic manipulation
* grasp planning.

---

# 7. Dataset Recording

### `zed_yolo_seg_off4.py`

Records a short sequence including:

* RGB video
* segmentation masks
* full point clouds

Files are saved frame-by-frame for later processing.

---

### `zed_yolo_seg_offv2.py`

Saves **segmented object point clouds** using YOLO masks.

Also records object statistics per frame.

---

### `zed_yolo_seg+pc_applev3.py`

Records a complete dataset including:

* RGB frames
* bounding boxes
* point clouds
* metadata (JSON)

Useful for:

* training machine learning models
* benchmarking algorithms.

---

# 8. Advanced Real-Time Integration

### `yolo11_zed.py`

Advanced pipeline integrating:

* YOLO inference in a **separate thread**
* ZED **object tracking**
* real-time visualization

Advantages:

* better performance
* reduced latency
* more scalable architecture.

---

# Which Script Should I Use?

For different tasks:

**Simple object detection**

```
zed_yolo.py
```

**Extract object point clouds**

```
zed_pc_yolo_v2.py
```

**Visualize segmented objects in 3D**

```
zed_seg_yolo.py
```

**Geometric analysis (curvature / radius)**

```
zed_yolo_seg+pc_apple.py
```

**Create datasets**

```
zed_yolo_seg_off4.py
```

**High-performance real-time pipeline**

```
yolo11_zed.py
```

---

# Important Notes

* YOLO segmentation masks must be **resized to match the ZED resolution** before applying them to the point cloud.
* Depth data may contain **NaN values**, which should always be filtered.
* Depth quality strongly affects **3D accuracy**.
* **Segmentation masks are usually more accurate than bounding boxes** for isolating objects.

---

# Project Goals

This toolkit enables:

* detecting objects in **3D space**
* extracting **clean object point clouds**
* estimating **geometric properties**
* recording **datasets for training**
* developing **robotic perception pipelines**


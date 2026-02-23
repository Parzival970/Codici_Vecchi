#!/usr/bin/env python3

import sys
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep

# Thread synchronization
lock = Lock()
run_signal = False
exit_signal = False
image_net = None
detections = []

def xywh2abcd(xywh, im_shape):
    """Convert YOLO bbox format to ZED format."""
    output = np.zeros((4, 2))

    x_min = xywh[0] - 0.5 * xywh[2]
    x_max = xywh[0] + 0.5 * xywh[2]
    y_min = xywh[1] - 0.5 * xywh[3]
    y_max = xywh[1] + 0.5 * xywh[3]

    output[0] = [x_min, y_min]
    output[1] = [x_max, y_min]
    output[2] = [x_min, y_max]
    output[3] = [x_max, y_max]

    return output

def detections_to_custom_box(detections, im0):
    """Convert YOLO detections to ZED CustomBox format."""
    output = []
    for det in detections:
        xywh = det.xywh[0]
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = int(det.cls[0].item())  # Convert to integer class ID
        obj.probability = float(det.conf[0].item())  # Convert to float
        obj.is_grounded = False
        output.append(obj)
    return output

def torch_thread(model_path, img_size, conf_thres=0.2, iou_thres=0.45):
    """Runs YOLO inference in a separate thread."""
    global image_net, exit_signal, run_signal, detections
    print("Initializing YOLOv11n model...")

    model = YOLO(model_path)

    while not exit_signal:
        if run_signal:
            lock.acquire()
            if image_net is not None:
                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)  # Ensure correct color format
                det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
                detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)

def main():
    """Main function to handle ZED camera + YOLO detection."""
    global image_net, exit_signal, run_signal, detections

    # Start YOLO inference thread
    capture_thread = Thread(target=torch_thread, kwargs={
        'model_path': opt.weights,
        'img_size': opt.img_size,
        'conf_thres': opt.conf_thres
    })
    capture_thread.start()

    print("Initializing ZED Camera...")
    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo:
        input_type.set_from_svo_file(opt.svo)

    # ZED initialization parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        exit(1)

    image_left_tmp = sl.Mat()

    print("ZED Camera Initialized")

    # Enable positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable object detection (Custom YOLO-based)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    while not exit_signal:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)

            # Convert ZED image to NumPy array
            image_net = image_left_tmp.get_data()
            image_net = np.copy(image_net)  # Ensure valid NumPy format

            # Ensure correct image format
            if image_net.shape[-1] == 4:
                image_net = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)

            lock.release()
            run_signal = True  # Trigger YOLO inference

            # Wait for detections
            while run_signal:
                sleep(0.001)

            lock.acquire()
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            # Display results
            image_show = np.copy(image_net)
            for obj in detections:
                x_min, y_min = int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1])
                x_max, y_max = int(obj.bounding_box_2d[3][0]), int(obj.bounding_box_2d[3][1])
                label = f"{obj.label}: {obj.probability:.2f}"
                cv2.rectangle(image_show, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image_show, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("ZED + YOLOv11n Detection", image_show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_signal = True
        else:
            exit_signal = True

    zed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov11n.pt', help='Path to YOLOv11n model')
    parser.add_argument('--svo', type=str, default=None, help='Optional SVO file for playback')
    parser.add_argument('--img_size', type=int, default=416, help='Inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='Object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()

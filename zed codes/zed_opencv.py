import sys
import numpy as np
import pyzed.sl as sl
import cv2

help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 

def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename):
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved:
        print("Done")
    else:
        print("Failed... Please check permissions to write on disk.")

def save_depth(zed, filename):
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved:
        print("Done")
    else:
        print("Failed... Please check permissions to write on disk.")

def save_sbs_image(zed, filename):
    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data().copy()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data().copy()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)
    cv2.imwrite(filename, sbs_image)

def process_key_event(zed, key):
    global mode_depth, mode_point_cloud, count_save, depth_format_ext, point_cloud_format_ext

    if key == ord('d'):
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == ord('n'):
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format:", depth_format_ext)
    elif key == ord('p'):
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == ord('m'):
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format:", point_cloud_format_ext)
    elif key == ord('h'):
        print(help_string)
    elif key == ord('s'):
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1

def print_help():
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")

def main():
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2:
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    print_help()

    runtime = sl.RuntimeParameters()

    # Reduce resolution to half
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width //= 2
    image_size.height //= 2

    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    key = ' '
    while key != ord('q'):
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            image_ocv = image_zed.get_data().copy()
            depth_image_ocv = depth_image_zed.get_data().copy()

            print("Image shape:", image_ocv.shape)
            print("Depth image shape:", depth_image_ocv.shape)

            if image_ocv is not None and depth_image_ocv is not None:
                # Convert BGRA (4 channels) to BGR (3 channels)
                image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

                # Normalize depth image to display correctly
                depth_gray = cv2.normalize(depth_image_ocv, None, 0, 255, cv2.NORM_MINMAX)
                depth_gray = np.uint8(depth_gray)

                # Ensure arrays are contiguous
                image_ocv = np.ascontiguousarray(image_ocv, dtype=np.uint8)
                depth_gray = np.ascontiguousarray(depth_gray, dtype=np.uint8)

                cv2.imshow("Image", image_ocv)
                cv2.imshow("Depth", depth_gray)
            else:
                print("Error: Images not retrieved properly")

            key = cv2.waitKey(10)
            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
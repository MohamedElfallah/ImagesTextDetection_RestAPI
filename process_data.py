import os
import numpy as np
LABELS_PATH = "./test_labels"
OUTPUT_PATH = "./yolov5/icdar_data/labels/"


def process(path, data_type="train"):
    labels = [f"gt_img_{i+1}.txt" for i in range(500)]
    for file_name in labels:
        file = open(os.path.join(LABELS_PATH, file_name), "r+")
        lines = file.readlines()
        yolo_format = []
        for line in lines:
            sp = line.split(",")
            sp[0] = sp[0].strip("ï»¿")
            coord = np.array(sp[:8], dtype=np.float32)
            x_right = max(coord[2], coord[4])
            x_left = min(coord[0], coord[6])
            y_top = min(coord[1], coord[3])
            y_bottom = max(coord[5], coord[7])
            width = x_right - x_left
            height = y_bottom - y_top
            x_center = x_left + width/2
            y_center = y_top + height/2
            width /= 1280.0
            height /= 720.0
            x_center /= 1280.0
            y_center /= 720.0
            yolo_format.append([0, x_center, y_center, width, height])
        yolo_format = np.array(yolo_format)
        np.savetxt(os.path.join(OUTPUT_PATH, data_type, file_name[3:]), yolo_format, fmt=[
                   "%d", "%f", "%f", "%f", "%f"])


if __name__ == "__main__":
    process(LABELS_PATH, data_type="validation")

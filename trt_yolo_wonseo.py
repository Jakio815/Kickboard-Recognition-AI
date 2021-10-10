"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import numpy as np

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        safe, on_board = wonseo(boxes, clss)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = draw(img, safe, on_board)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

def wonseo(boxes, clss):
    person_index = [idx for idx, value in enumerate(clss) if value == 0]
    kick_index = [idx for idx, value in enumerate(clss) if value == 1]
    helmet_index = [idx for idx, value in enumerate(clss) if value == 2]

    on_board = []
    safe = []
    
    for i in kick_index:
        person = 0
        safety = 0
        x_center = (boxes[i][0] + boxes[i][2])//2
        y_center = (boxes[i][1] + boxes[i][3])//2
        kickboard_size = (boxes[i][3] - boxes[i][1])//2 + (boxes[i][2] - boxes[i][0])
        for j in person_index:
            person_x_center =(boxes[j][0] + boxes[j][2])//2
            person_y_center =(boxes[j][1] + boxes[j][3])//2
            
            if abs(x_center - person_x_center) + abs(y_center - person_y_center) <= kickboard_size:
                person += 1
                
                for k in helmet_index:
                    if IoU(boxes[j], boxes[k]) > 0.03:
                        safe.append([[x_center, y_center], [person_x_center, person_y_center], [(boxes[k][0] + boxes[k][2])//2, (boxes[k][1] + boxes[k][3])//2]])
                        safety += 1
                        break
                
                if safety == 0 and person == 1:
                    on_board.append([[x_center, y_center], [person_x_center, person_y_center]])
                elif safety == 0 and person == 2:
                    on_board[-1].append([person_x_center, person_y_center])
    return safe, on_board

            

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

# def draw(frame, safe):
#     if len(safe) >=1:   
#         for safes in safe:
#             cv2.polylines(frame, np.array([safes]), False, (255,0,0))
#             cv2.putText(frame, text="Safe", org=safes[2], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 0, 255), thickness=3)
#     return frame

def draw(frame, safe, on_board):
    if len(safe) >=1:   
        for safes in safe:
            # print(safes)
            # print(safes[2])
            cv2.polylines(frame, np.array([safes]), False, (255,0,0))
            cv2.putText(frame, text="Safe", org=(safes[2][0], safes[2][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0, 255, 0), thickness=3)
    if len(on_board) >=1:   
        for on_boards in on_board:
            cv2.polylines(frame, np.array([on_boards]), False, (255,0,0))
            # print(on_boards)
            # print(len(on_boards))
            # print((on_boards[1][0], on_boards[1][1]))
            cv2.putText(frame, text="%i number of people"%(len(on_boards)-1), org=(on_boards[1][0], on_boards[1][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=3)
    return frame


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict2(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

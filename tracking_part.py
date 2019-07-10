######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.

# Import packages
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

# from grabscreen import grab_screen

# original import packages
from my_changed_utils import visualization_utils as vis_util
from my_changed_utils.Vehicle_counter import VehicleCounter
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import math
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
# VIDEO_NAME = 'VID_20190517_191713.mp4'
# VIDEO_NAME = 'NFS Underground 2 2019-07-04 00-07-48.mp4'
# VIDEO_NAME = 'west.mp4'
# VIDEO_NAME = 'infra.mp4'
VIDEO_NAME = 'day.avi'
# VIDEO_NAME = 'night.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 4

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# print("#################################  here is the thing: {} {}  #################################".format(
#     type(detection_boxes), detection_boxes.shape))
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')



# ========================================================================================================================================== #
                                            # Getting Dimensions and other properties of frame
cap = cv2.VideoCapture(VIDEO_NAME)
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
length = int(width)
height = int(height)
cap.release()
# ========================================================================================================================================== #
                                                            # Specific Colours
DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)
                                                            # Divider boundaries
DIVIDER1 = (DIVIDER1_A, DIVIDER1_B) = ((length // 2 + 200 + 215, height//2 - 30 + 60),(length // 2 + 200 + 435, height//2 - 10 + 60) )
DIVIDER2 = (DIVIDER2_A, DIVIDER2_B) = ((length // 2 + 200 - 50, height//2 + 10 - 4), (length // 2, 350))
DIVIDER3 = (DIVIDER3_A, DIVIDER3_B) = ((length // 2 + 200 + 215, height//2 - 10 + 45),(length // 2 + 200 - 50, height//2 + 10 - 4))
DIVIDER4 = (DIVIDER4_A, DIVIDER4_B) = ((length // 2 + 200 - 340, height//2 - 20), (length // 2 + 200 - 350 + 120, height//2 - 30))
# ========================================================================================================================================== #
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = math.floor(x + x1)
    cy = math.floor(y + y1)

    return (cx, cy)

# ============================================================================ #
                            # Detecting Contours Function
# ============================================================================ #
# def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=35):
#     # finding external contours
#     # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

#     # filtering by with, height 
#     contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     final_contour_list = []
#     centroid_aftercal = []
#     my_centroid_aftercal = []
#     for (i, contour) in enumerate(contours):
#         # getting the bounding box dimensions
#         (x, y, w, h) = cv2.boundingRect(contour)
#         # Finding area of the contour
#         ar = cv2.contourArea(contour)
#         # to find if the contour is valid or not
#         contour_valid = (hierarchy[0, i, 3] == -1) and (ar > 550) and ((w > h and (w/h < 3.5)) or (h/w < 3.5)) and (y>230)

#         if not contour_valid:
#             continue
#         # getting the centroid of the contour
#         centroid = get_centroid(x, y, w, h)
#         # appending all the contours to a final list
#         final_contour_list.append(((x, y, w, h), centroid))
#     # combines the nearby centroids
#     centroid_combined = combined_nearby_centroid(final_contour_list)

#     for entry in centroid_combined:
#         tempx = []
#         tempy = []
#         temp_cnt_x = []
#         temp_cnt_y = []
#         temp_cnt_w = []
#         temp_cnt_h = []
#         for centroid in entry:
#             tempx.append(centroid[1][0])
#             tempy.append(centroid[1][1])
#             temp_cnt_x.append(centroid[0][0])
#             temp_cnt_y.append(centroid[0][1])
#             temp_cnt_w.append(centroid[0][2])
#             temp_cnt_h.append(centroid[0][3])
#         x, y, w, h = sum(temp_cnt_x)//len(temp_cnt_x), sum(temp_cnt_y)//len(temp_cnt_y), sum(temp_cnt_w)//len(temp_cnt_w), sum(temp_cnt_h)//len(temp_cnt_h)
#         cent_x, cent_y = sum(tempx) // len(tempx), sum(tempy) // len(tempy)
#         my_centroid_aftercal.append(((x,y,w,h),(cent_x,cent_y)))
#     return my_centroid_aftercal,final_contour_list

# ============================================================================ #
                                # Processing Function
# ============================================================================ #
def process_frame(frame,car_counter):
    score_thresh = 0.5
    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(frame, DIVIDER1_A, DIVIDER1_B, DIVIDER_COLOUR, 1)
    cv2.line(frame, DIVIDER2_A, DIVIDER2_B, DIVIDER_COLOUR, 1)
    cv2.line(frame, DIVIDER3_A, DIVIDER3_B, DIVIDER_COLOUR, 1)
    cv2.line(frame, DIVIDER4_A, DIVIDER4_B, DIVIDER_COLOUR, 1)

    # Drawing circles at the endpoints of the dividers
    cv2.circle(frame, DIVIDER3_A, 5, (255,0,0),-1)
    cv2.circle(frame, DIVIDER3_B, 5, (0,255,0),-1)
    cv2.circle(frame, DIVIDER1_A, 5, (0,0,255),-1)
    cv2.circle(frame, DIVIDER2_B, 5, (150,15,155),-1)

    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # print("#################################  here is the thing: {} {}  #################################".format(
    #     type(boxes), boxes.shape))
    # Draw the results of the detection (aka 'visulaize the results')

    # print(len(boxes[0]), len(scores[0]), len(classes[0]), num)
    # Draw the results of the detection (aka 'visulaize the results')
    # print(boxes[0])
    # A box here is Ymin,Xmin,Ymax,Xmax in normalized form so multiply Y with height and X with width
    i = 0
    item = ['car', 'bus', 'person', 'truck', 'traffic_sign',
            'traffic_light', 'bike', 'rider', 'motor']
    lst = []
    for b, s, c in zip(boxes[0], scores[0], classes[0]):
        if(s >= score_thresh):
            Ymin = b[0]*height
            Xmin = b[1]*width
            Ymax = b[2]*height
            Xmax = b[3]*width
            w = Xmax - Xmin
            h = Ymax - Ymin
            centroid = get_centroid(Xmin, Ymin, w, h)
            lst.append((b,s,c,centroid))
            i+=1
            cv2.circle(frame,centroid, 5, (0,0,255), -1)
            # cv2.rectangle(image, (math.ceil(b[1]*1280), math.ceil(b[0]*720)),
            #               (math.ceil(b[3]*1280), math.ceil(b[2]*720)), (255, 255, 0), 3)
    print('nums are: ', i)

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=score_thresh)
    # detected_contours,boxes = detect_vehicles(filtered_mask)
    # updating the count
    car_counter.update_count(lst, frame)
    # getting the list of vehicles
    v = car_counter.vehicles
    
    for veh in v:
        pos = veh.positions[-1]
        veh_id = veh.id
        cv2.putText(processed, ("id:%02d" % veh_id), pos
            , cv2.FONT_HERSHEY_PLAIN, 0.7, (55, 255, 55), 2)

    # for (i, box) in enumerate(boxes):
    #     # Mark the bounding box on the processed frame
    #     x = box[0][0]
    #     y = box[0][1]
    #     w = box[0][2]
    #     h = box[0][3]
    #     cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame



# Open video file

def main():
    video = cv2.VideoCapture(PATH_TO_VIDEO)
    my_frame_no = 0
    offset = 2000
    start = time.time()
    car_counter = None
    while(video.isOpened()):
        my_frame_no += 1
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        if(my_frame_no < offset):
            print(my_frame_no, end='\t')
            continue
        if car_counter is None:
            car_counter = VehicleCounter(frame.shape[:2], DIVIDER1, DIVIDER2, DIVIDER3, DIVIDER4)
        processed = process_frame(frame,car_counter)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', processed)
        # end = time.time()
        # passed_time = end-start
        # print('frames till now : {}, offset is : {}, time elapsed is {}, fps is : {}'.format(
        #     my_frame_no, offset, passed_time, (my_frame_no-offset)/passed_time))
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import math
import os
import random
import numpy as np
# import skvideo.io
# import matplotlib.pyplot as plt
# import utils
# without this some strange errors happen
# cv2.ocl.setUseOpenCL(False)
random.seed(123)

# ============================================================================
IMAGE_DIR = "./out"
VIDEO_SOURCE = "../../traffic_videos/NIGHT_TIME/vid1.avi"
SHAPE = (720, 1280)  # HxW
# ============================================================================


# def train_bg_subtractor(inst, cap, num=500):
#     '''
#         BG substractor need process some amount of frames to start giving result
#     '''
#     print ('Training BG Subtractor...')
#     i = 0
#     clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
#     for frame in cap.read():
#         if(type(frame)==bool):
#             continue
#         print(frame.shape)
#         print('frame one',frame.shape)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         hsv_planes = cv2.split(hsv)
#         cv2.imshow("v component",hsv_planes[2])
#         hsv_planes[2] = cv2.threshold(hsv_planes[2],140,255,cv2.THRESH_TOZERO_INV)
#         cv2.imshow("thresholded v component",cv2.UMat(imgUMat) )
#         hsv_new = cv2.merge(hsv_planes)
#         out = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
#         inst.apply(out, None, 0.001)
#         i += 1

def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    for frame in cap.read():
        # print(frame)
        inst.apply(frame, None, 0.001)
        i += 1
        if i >= num:
            return cap

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


def filter_mask(img):
    # Making the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 6))
    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closing',closing)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('Opening',opening)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=9)
    _, mask = cv2.threshold(dilation, 240, 255, cv2.THRESH_BINARY)
    cv2.imshow('Dilation and Thresholding',mask)
    # Applying Gaussian Blur
    mask = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)
    cv2.imshow('Gauss Mask',mask)
    _, mask = cv2.threshold(mask, 35, 255, cv2.THRESH_BINARY)
    # Applying Median Blur
    mask = cv2.medianBlur(mask, 5)
    cv2.imshow('Median mask',mask)
    # Final Mask
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    
    return mask

def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=35):
    # finding external contours
    # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # filtering by with, height 
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contour_list = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        ar = cv2.contourArea(contour)
        contour_valid = (ar > 550) and ((w > h and (w/h < 3.5)) or (h/w < 3.5))

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        final_contour_list.append(((x, y, w, h), centroid))
    return final_contour_list

def main():

    # creating MOG bg subtractor with 500 frames in cache
    # and shadow detction
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50, detectShadows=True)

    # Set up image source
    # You can use also CV2, for some reason it not working for me
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # skipping 500 frames to train bg subtractor
    # train_bg_subtractor(bg_subtractor, cap, num=500)

    frame_number = -1
    while True:
        # Reading one frame
        _, frame = cap.read()
        frame_number += 1
        # utils.save_frame(frame, "./out/frame_%04d.png" % frame_number)
        foreGround_mask = bg_subtractor.apply(frame)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
        filtered_mask = filter_mask(foreGround_mask)
        # # utils.save_frame(frame, "./out/fg_mask_%04d.png" % frame_number)
        detected_contours = detect_vehicles(filtered_mask)
        for contour in detected_contours:
            x = contour[0][0]
            y = contour[0][1]
            w = contour[0][2]
            h = contour[0][3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        try:
            cv2.imshow("Frame", frame)
            cv2.imshow("mask", filtered_mask)
        except:
            pass

        key = cv2.waitKey(3)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
# ============================================================================

if __name__ == "__main__":
    main()

import cv2
import math
import os
import random
import numpy as np
from Vehicle_counter import VehicleCounter
# import skvideo.io
# import matplotlib.pyplot as plt
# import utils
# without this some strange errors happen
# cv2.ocl.setUseOpenCL(False)
random.seed(123)

# ============================================================================ #
                                # Video Source
IMAGE_DIR = "./out"
VIDEO_SOURCE = "../../traffic_videos/NIGHT_TIME/vid1.avi"
SHAPE = (720, 1280)  # HxW
# ========================================================================================================================================== #
                                            # Getting Dimensions and other properties of frame
cap = cv2.VideoCapture(VIDEO_SOURCE)
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
DIVIDER1 = (DIVIDER1_A, DIVIDER1_B) = ((length // 2 + 200 + 435, height//2 - 10 + 120), (length // 2 + 200 + 215, height//2 - 10 + 110))
DIVIDER2 = (DIVIDER2_A, DIVIDER2_B) = ((length // 2 + 200 - 50, height//2 + 10 - 4), (length // 2, 350))
DIVIDER3 = (DIVIDER3_A, DIVIDER3_B) = ((length // 2 + 200 + 215, height//2 - 10 + 45),(length // 2 + 200 - 50, height//2 + 10 - 4))
DIVIDER4 = (DIVIDER4_A, DIVIDER4_B) = ((length // 2 + 200 + 115, height//2 - 10 + 110), (length // 2 - 180, height//2 - 10 + 360))
# ========================================================================================================================================== #

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

# ============================================================================ #
                                # Filtering Function
# ============================================================================ #
def filter_mask(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    cv2.imshow('lap',laplacian)
    # Making the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 6))
    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closing',closing)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('Opening',opening)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=9)
    _, mask = cv2.threshold(dilation, 200, 255, cv2.THRESH_BINARY)
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
# ============================================================================ #
def combined_nearby_centroid(contour_list):
    centroid_pool = [x[1] for x in contour_list]
    centroid_combined = []
    my_centroid_pool = contour_list[:]
    my_centroid_combined = []
    i = 0
    for (i, centroid) in enumerate(my_centroid_pool):
        flag = 0
        for entry in my_centroid_combined:
            if centroid in entry:
                flag = 1
                break
        if flag == 0:
            my_centroid_combined.append([centroid])
        for j in range(i, len(my_centroid_pool)):
            if abs(centroid[1][0] - my_centroid_pool[j][1][0]) < 100 and abs(centroid[1][1] - my_centroid_pool[j][1][1]) < 40:    
                for entry in my_centroid_combined:
                    if centroid in entry and my_centroid_pool[j] not in entry:
                        entry.append(my_centroid_pool[j])

    for (i, centroid) in enumerate(centroid_pool):
        flag = 0
        for entry in centroid_combined:
            if centroid in entry:
                flag = 1
                break
        if flag == 0:
            centroid_combined.append([centroid])
        for j in range(i, len(centroid_pool)):
            if abs(centroid[0] - centroid_pool[j][0]) < 100 and abs(centroid[1] - centroid_pool[j][1]) < 40:    
                for entry in centroid_combined:
                    if centroid in entry and centroid_pool[j] not in entry:
                        entry.append(centroid_pool[j])
    # my_new_centroid_combined = [x[1] for x in my_centroid_combined]
    # print("my: ",np.array(my_centroid_combined))
    # print("new: ",np.array(centroid_combined))
    # exit()
    return my_centroid_combined
# ============================================================================ #
                            # Detecting Contours Function
# ============================================================================ #
def detect_vehicles(fg_mask, min_contour_width=35, min_contour_height=35):
    # finding external contours
    # contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # filtering by with, height 
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contour_list = []
    centroid_aftercal = []
    my_centroid_aftercal = []
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        ar = cv2.contourArea(contour)
        contour_valid = (hierarchy[0, i, 3] == -1) and (ar > 550) and ((w > h and (w/h < 3.5)) or (h/w < 3.5)) and (y>230)

        if not contour_valid:
            continue

        centroid = get_centroid(x, y, w, h)

        final_contour_list.append(((x, y, w, h), centroid))

    centroid_combined = combined_nearby_centroid(final_contour_list)

    for entry in centroid_combined:
        tempx = []
        tempy = []
        temp_cnt_x = []
        temp_cnt_y = []
        temp_cnt_w = []
        temp_cnt_h = []
        for centroid in entry:
            tempx.append(centroid[1][0])
            tempy.append(centroid[1][1])
            temp_cnt_x.append(centroid[0][0])
            temp_cnt_y.append(centroid[0][1])
            temp_cnt_w.append(centroid[0][2])
            temp_cnt_h.append(centroid[0][3])
        x, y, w, h = sum(temp_cnt_x)//len(temp_cnt_x), sum(temp_cnt_y)//len(temp_cnt_y), sum(temp_cnt_w)//len(temp_cnt_w), sum(temp_cnt_h)//len(temp_cnt_h)
        cent_x, cent_y = sum(tempx) // len(tempx), sum(tempy) // len(tempy)
        my_centroid_aftercal.append(((x,y,w,h),(cent_x,cent_y)))

    # for entry in centroid_combined:
    #     tempx = []
    #     tempy = []
    #     for centroid in entry:
    #         tempx.append(centroid[0])
    #         tempy.append(centroid[1])
    #     centroid_aftercal.append((sum(tempx) // len(tempx), sum(tempy) // len(tempy)))
    return my_centroid_aftercal,final_contour_list

# ============================================================================ #
                                # Processing Function
# ============================================================================ #
def process_frame(frame, bg_subtractor,car_counter):

    # Create a copy of source frame to draw into
    processed = frame.copy()

    # Draw dividing line -- we count cars as they cross this line.
    cv2.line(processed, DIVIDER1_A, DIVIDER1_B, DIVIDER_COLOUR, 1)
    cv2.line(processed, DIVIDER2_A, DIVIDER2_B, DIVIDER_COLOUR, 1)
    cv2.line(processed, DIVIDER3_A, DIVIDER3_B, DIVIDER_COLOUR, 1)
    cv2.line(processed, DIVIDER4_A, DIVIDER4_B, DIVIDER_COLOUR, 1)

    # Drawing circles at the endpoints of the dividers
    cv2.circle(processed, DIVIDER3_A, 5, (255,0,0),-1)
    cv2.circle(processed, DIVIDER3_B, 5, (0,255,0),-1)
    cv2.circle(processed, DIVIDER1_A, 5, (0,0,255),-1)
    cv2.circle(processed, DIVIDER2_B, 5, (150,15,155),-1)

    # Remove the background
    foreGround_mask = bg_subtractor.apply(frame, None, 0.01)
    
    # Filter the foreGround
    filtered_mask = filter_mask(foreGround_mask)
    # utils.save_frame(frame, "./out/fg_mask_%04d.png" % frame_number)
    detected_contours,boxes = detect_vehicles(filtered_mask)

    car_counter.update_count(detected_contours, processed)
    v = car_counter.vehicles
    
    for veh in v:
        pos = veh.positions[-1][1]
        veh_id = veh.id
        cv2.putText(processed, ("id:%02d" % veh_id), pos
            , cv2.FONT_HERSHEY_PLAIN, 0.7, (55, 255, 55), 2)

    for (i, box) in enumerate(boxes):
        centroid = box
        # print(type(centroid))
        # print(centroid)
        # Mark the bounding box and the centroid on the processed frame
        # NB: Fixed the off-by one in the bottom right corner
        # cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        x = box[0][0]
        y = box[0][1]
        w = box[0][2]
        h = box[0][3]
        cv2.rectangle(processed, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.circle(processed, centroid, 20, CENTROID_COLOUR, -1)


    return processed

# ============================================================================ #
                                # Main Function
# ============================================================================ #
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
    car_counter = None
    frame_number = -1
    while True:
        # Reading one frame
        _, frame = cap.read()
        frame_number += 1
        if car_counter is None:
                # We do this here, so that we can initialize with actual frame size
                car_counter = VehicleCounter(frame.shape[:2], DIVIDER1, DIVIDER2, DIVIDER3, DIVIDER4)
        processed = process_frame(frame,bg_subtractor,car_counter)
        try:
            cv2.imshow("Frame", frame)
            cv2.imshow("processed", processed)
        except:
            pass

        key = cv2.waitKey(3)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================================ #
if __name__ == "__main__":
    main()

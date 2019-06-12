import cv2
import numpy as np
# import imutils

# Reading The video
# cap = cv2.VideoCapture("../../traffic_videos/NIGHT_TIME/vid1.avi")

# Initializing The subtractor
# subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=28)
# cv2.ocl.setUseOpenCL(False)

subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, detectShadows=True)




def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print ('Training BG Subtractor...')
    i = 0
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    while(i<num):
        _,frame = cap.read()
        # print(frame.shape)
        # print('frame one',frame)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)


        lab_planes[0] = clahe.apply(lab_planes[0])
        # lab_planes[1] = clahe.apply(lab_planes[1])
        lab_planes[2] = clahe.apply(lab_planes[2])

        lab = cv2.merge(lab_planes)

        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        inst.apply(out, None, 0.001)
        i += 1
# train_bg_subtractor(subtractor, cap, num=500)
# cap = skvideo.io.vreader("../../traffic_videos/NIGHT_TIME/vid1.avi")
cap = cv2.VideoCapture("../../traffic_videos/NIGHT_TIME/vid1.avi")


while True:
    # Reading one frame
    _, frame = cap.read()


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

    lab_planes[0] = clahe.apply(lab_planes[0])
    # lab_planes[1] = clahe.apply(lab_planes[1])
    lab_planes[2] = clahe.apply(lab_planes[2])
    # print(type(lab_planes[0]))
    # print(lab_planes.__len__())

    # _, lab_planes[0] = cv2.threshold(lab_planes[0], 15, 255, cv2.THRESH_BINARY)
    # _, lab_planes[1] = cv2.threshold(lab_planes[1], 15, 255, cv2.THRESH_BINARY)
    # _, lab_planes[2] = cv2.threshold(lab_planes[2], 15, 255, cv2.THRESH_BINARY)

    lab = cv2.merge(lab_planes)

    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Initializing CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Converting colourspace
    # out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying CLAHE
    # out = clahe.apply(out)
    # cv2.imshow("out", out)
    # if(i==2048):
    #     cv2.imwrite("latest/clahe.jpg",out)
    # out = cv2.equalizeHist(out)
    # applying b/g substraction

    hsv = cv2.cvtColor(out,cv2.COLOR_BGR2HSV)

    components = cv2.split(hsv)
    # print('length is ',components.__len__())
    # cv2.imshow("H component",components[0])
    # cv2.imshow("S component",components[1])


    cv2.imshow("v channel",components[2])
    components[2] = cv2.medianBlur(components[2],7)

    cv2.imshow("gaussian v channel",components[2])

    _, components[2] = cv2.threshold(components[2], 60, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("threshold v channel",components[2])
    components[2] = cv2.dilate(components[2], kernel, iterations=2)

    # components[2] = components[2] - new_V
    # cv2.imshow("V component",components[2])
    hsv = cv2.merge(components)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Applying Background Subtraction
    mask = subtractor.apply(out)
    print('second type',type(mask))
    # if(i==2048):
    #     cv2.imwrite("latest/after_subtraction.jpg",mask)

    # Fill any small holes
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)
    _, mask = cv2.threshold(dilation, 254, 255, cv2.THRESH_BINARY)

    cv2.imshow('nOT mINE',mask)
    # Applying Gaussian Blur
    mask = cv2.GaussianBlur(mask, (13, 9), cv2.BORDER_DEFAULT)
    cv2.imshow('Gauss Mask',mask)
    # kernel2 = np.ones((3, 3), np.uint8)
    # dilation = cv2.dilate(mask, kernel2, iterations=1)
    # mask = cv2.medianBlur(mask, 7)

    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    # Applying Median Blur
    mask = cv2.medianBlur(mask, 11)
    cv2.imshow('Median mask',mask)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel2, iterations=1)
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # mask = cv2.medianBlur(mask, 5)
    # mask = cv2.medianBlur(mask, 5)
    # if(i==2048):
    #     cv2.imwrite("latest/Blurred.jpg",mask)

    # binarization of the frame
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    # _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    # mask = cv2.GaussianBlur(mask,(17,17),0)
    # mask = cv2.medianBlur(mask,5)
    # _,mask = cv2.threshold(mask,45,255,cv2.THRESH_BINARY)
    # mask = cv2.GaussianBlur(mask,(9,9),0)
    # mask = cv2.medianBlur(mask,5)
    # _,mask = cv2.threshold(mask,45,255,cv2.THRESH_BINARY)
    # if(i==2048):
    #     cv2.imwrite("latest/After_thresholding.jpg",mask)
    # except:
    # pass

    # extracting the contours from the image
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(
    #     mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = contours
    # print(cnts.__len__())
    # cnts = cnts[0]
    # cnt = cnts[4]
    # print(cnts)
    # try:
    #     # cv2.drawContours(frame, [cnts], 0, (0,255,0), 3)
    # except:
    #     pass

    # iterating over all the contours present in the frame
    for i in range(len(cnts)):
        c = cnts[i]
        # print(c)
        # print(cv2.contourArea(c))

        # finding the area of the bounding rectangle
        ar = cv2.contourArea(c)

        # finding the coordinates of the bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # checking if the area is greater than some threshold
        if(ar > 650):
            if(w > h and (w/h < 3.5)) or (h/w < 3.5):
                # print("w/h is : ", w/h)
                # print("x is : {}, y is : {}, w is : {}, h is : {}".format(x, y, w, h))
                # cv2.circle(frame, (x, y), 20, 2)
                # print("h/w is : ", h/w)
                # print(cv2.contourArea(c))

                # this function draws the bounding box arround the contour
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     # try:
    #     # cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    #     # except:
    #     # print("here : ",c)
    #     # print("next")
    #     # pass
    # try:
    #     cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)
    # except:
        # print("here : ",c)
        # print("next")
        # pass
    # print("frame")
    # show the output image
    try:
        cv2.imshow("Frame", frame)
        cv2.imshow("mask", mask)
    except:
        pass

    key = cv2.waitKey(3)
    if key == 27:
        break
    # if(i==2048):
    #     break

cap.release()
cv2.destroyAllWindows()
# press escape to exit in middle of video

# def histoFn(img_path):
#     # img_path = "NIGHT_TIME_1_frame_random_92.jpg"
#     img = cv2.imread(img_path,1)
#
#     # cv2.imshow("ksdc",img)
#     try:
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     except:
#         list_failed.append(img_path)
#         return
#     R, G, B = cv2.split(img)
#
#     output1_R = cv2.equalizeHist(R)
#     output1_G = cv2.equalizeHist(G)
#     output1_B = cv2.equalizeHist(B)
#
#     output1 = cv2.merge((output1_B,output1_G,output1_R))
#     # clahe = cv2.createCLAHE()
#     clahe = cv2.createCLAHE(clipLimit = 2.0,tileGridSize = (8,8))
#
#     output2_R = clahe.apply(R)
#     output2_G = clahe.apply(G)
#     output2_B = clahe.apply(B)
#
#     output2 = cv2.merge((output2_B,output2_G,output2_R))
#
#     p = img_path.split("orig/")
#     adjusted_histo_path = ''
#     clahe_path = p[0] + "CLAHE/" + "clahe_" + p[-1]
#     adjusted_histo_path = p[0] + "Adjusted_histogram/" + "adjusted_histo_" + p[-1]
#     output = [output1,output2]
#     output_paths = []
#     output_paths.append(adjusted_histo_path)
#     output_paths.append(clahe_path)
#     # titles = ['adjusted_histo.jpg','CLAHE.jpg']
#     # print("p is ")
#     # print(p)
#     # print(clahe_path,"\t",adjusted_histo_path,end="\n")
#
#     for i in range(2):
#         # plt.subplot(1,3,i+1)
#         plt.imshow(output_paths[i],output[i])
#         # cv2.imwrite(output_paths[i],output[i])
#         # plt.title(titles[i])
#         # plt.xticks([])
#         # plt.yticks([])
#     # plt.show()

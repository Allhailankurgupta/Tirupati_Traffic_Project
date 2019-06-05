import cv2
import numpy as np
import imutils
cap = cv2.VideoCapture("../traffic_videos/NIGHT_TIME/vid1.avi")

subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=28)
i = 0
l = 0
while True:
    _, frame = cap.read()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = clahe.apply(out)
    cv2.imshow("out", out)
    # if(i==2048):
    #     cv2.imwrite("latest/clahe.jpg",out)
    # out = cv2.equalizeHist(frame_2)
    mask = subtractor.apply(out)
    # if(i==2048):
    #     cv2.imwrite("latest/after_subtraction.jpg",mask)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # mask = cv2.medianBlur(mask,5)
    # if(i==2048):
    #     cv2.imwrite("latest/Blurred.jpg",mask)

    # _,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
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
    # im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    # cnt = cnts[4]
    # print(cnts)
    if l == 2:
        break
    # try:
    #     # cv2.drawContours(frame, [cnts], 0, (0,255,0), 3)
    # except:
    #     pass
    for i in range(len(cnts)):
        c = cnts[i]
        if(cv2.contourArea(c) < 1450):
            if(cv2.contourArea(c) > 1000):
                print(cv2.contourArea(c))
            continue
        # M = cv2.moments(c)
        # c = c.astype("float")

        # c = c.astype("int")
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # try:
        # cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        # except:
        # print("here : ",c)
        # print("next")
        # pass
    # try:
    #     cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)
    # except:
        # print("here : ",c)
        # print("next")
        # pass
    print("frame")
    # show the output image
    try:
        cv2.imshow("Frame", frame)
        cv2.imshow("mask", mask)
    except:
        pass

    key = cv2.waitKey(3000)
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

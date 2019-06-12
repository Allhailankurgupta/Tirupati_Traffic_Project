# # Initializing The subtractor
subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

# def train_bg_subtractor(inst, cap, num=500):
#     '''
#         BG substractor need process some amount of frames to start giving result
#     '''
#     print ('Training BG Subtractor...')
#     i = 0
#     clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
#     for frame in cap:
#         # print(frame.shape)
#         # print('frame one',frame)
#         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#         lab_planes = cv2.split(lab)


#         lab_planes[0] = clahe.apply(lab_planes[0])
#         # lab_planes[1] = clahe.apply(lab_planes[1])
#         lab_planes[2] = clahe.apply(lab_planes[2])

#         lab = cv2.merge(lab_planes)

#         out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#         inst.apply(out, None, 0.001)
#         i += 1
#         if i >= num:
#             return cap
        
# train_bg_subtractor(subtractor, cap, num=500)
# cap = skvideo.io.vreader("../../traffic_videos/NIGHT_TIME/vid1.avi")
cap = cv2.VideoCapture("../../traffic_videos/NIGHT_TIME/vid1.avi")
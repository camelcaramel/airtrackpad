import numpy as np
from cv2 import cv2
cap = cv2.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=300,
                      qualityLevel=0.01,
                      minDistance=30,
                      blockSize=14)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=0,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_diff = old_gray
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
# lower_blue = np.array([0, 50, 50])
# upper_blue = np.array([20, 255, 255])
# color_mask = cv2.inRange(hsv, lower_blue, upper_blue)
# res = cv2.bitwise_and(frame, frame, mask=mask)
W = old_frame.shape[0]
H = old_frame.shape[1]
number_mask = [[(max(w, h)) for w in range(W)] for h in range(H)]
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.bitwise_xor(frame_gray, old_gray) #####
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1[st == 1] is not None:
        good_new = p1[st == 1]
    if p0[st == 1] is not None:
        good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    # img = cv2.add(frame,mask)
    img1 = frame

    cur_diff = cv2.bitwise_xor(frame_gray, old_gray)
    # diff = diff * number_mask
    # frame = cv2.circle(frame,(diff.mean(axis=1), diff.mean(axis=0)),8,0,-1)
    # ret, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY) ###이진화
    # contours, image = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if cur_diff.any():
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_diff, cur_diff, p0, None, **lk_params)
        # Select good points
        if p1[st == 1] is not None:
            good_new = p1[st == 1]
        if p0[st == 1] is not None:
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame1 = cv2.circle(old_diff, (a, b), 5, color[i].tolist(), -1)

    diff = frame1

    cv2.imshow('gray', diff)
    cv2.imshow('frame', img1)
    k = cv2.waitKey(30) & 0xff
    if cur_diff.any():
        old_diff = cur_diff.copy()
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    # print(len(err))
cv2.destroyAllWindows()

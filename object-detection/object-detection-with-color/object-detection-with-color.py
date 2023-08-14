"""
We will learn how to detect objects in certain colors with the contour finding method.

Contours can simply be described as a curve connecting continuous points of the same color or density.

Contours are a useful tool for shape analysis and object detection and recognition.
"""

"""
ONE OBJECT 

import cv2
import numpy as np
from collections import deque # to store the center of the detected object

buffer_size = 16 # dque size
pts = deque(maxlen=buffer_size) # objects center point

# blue color,  interval HSV -> hue, satisfaction, brightness
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(3,960) # weight
cap.set(4, 480) # height

while True:
    success, imgOriginal = cap.read()

    if success:
        #blurr
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) # kernel size 11, sd 0
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        # mask for blue
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image", mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("mask + erosion + dilate Image", mask)

        # contour
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None # object center

        if len(contours) > 0 :
            # get max contour
            c = max(contours, key = cv2.contourArea)

            # convert contour to rectangle
            rect = cv2.minAreaRect(c)
            ((x,y), (width,height), rotation) = rect
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)

            # create a box with the width and height values obtained from the rectangle
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # moment -> average of image pixel densities
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            # draw contour
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)

            # draw point to center
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)

            # print the data
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)


        # tracking algorithm
        pts.append(center)
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(imgOriginal, pts[i-1], pts[i], (0,255,255),3)


        cv2.imshow("orijinal", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"): break"""

import cv2
import numpy as np
from collections import deque # to store the center of the detected object

buffer_size = 16 # dque size
pts = deque(maxlen=buffer_size) # objects center point

# blue color,  interval HSV -> hue, satisfaction, brightness
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

cap = cv2.VideoCapture(0)
cap.set(3,960) # weight
cap.set(4, 480) # height

while True:
    success, imgOriginal = cap.read()

    if success:
        #blurr
        blurred = cv2.GaussianBlur(imgOriginal, (11,11), 0) # kernel size 11, 11, sd 0
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        # mask for blue
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image", mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("mask + erosion + dilate Image", mask)

        # contour
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None # object center

        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter out small contours
                continue

            # convert contour to rectangle
            rect = cv2.minAreaRect(contour)
            ((x,y), (width,height), rotation) = rect
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)

            # create a box with the width and height values obtained from the rectangle
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # moment -> average of image pixel densities
            M = cv2.moments(contour)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            # draw contour
            cv2.drawContours(imgOriginal, [box], 0, (0,255,255),2)

            # draw point to center
            cv2.circle(imgOriginal, center, 5, (255,0,255),-1)

            # print the data
            cv2.putText(imgOriginal, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)


        # tracking algorithm
        pts.append(center)
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            cv2.line(imgOriginal, pts[i-1], pts[i], (0,255,255),3)


        cv2.imshow("orijinal", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"): break
"""
Mean Shift, a clustering algorithm that iteratively assigns data points to clusters by shifting points towards the mode, which can be defined as the highest data point density.

Hence, it is also known as a mode-seeking algorithm.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret == False:
    print("warning")

# detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w, h) = tuple(face_rects[0]) # first
track_window = (face_x, face_y, w, h) # mean-shift algorithm input

roi = frame[face_y:face_y+h, face_x:face_x+w] # roi = face

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180],[0,180]) # hist required for tracking, start 0 ending 180, range 0,180
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# stopping criteria for tracking
# count = maximum item 
# epsilon = change
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1) # 5 count 1 epsilon
while True:
    ret, frame =cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # use histgoram for finding some image
        # pixel compration
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x,y,w,h = track_window

        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 0)

        cv2.imshow("tracking", img2)

        if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.relase() 
cv2.destroyAllWindows()
















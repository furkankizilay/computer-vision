# https://developers.google.com/mediapipe/solutions/vision/face_detector/

import cv2
import mediapipe as mp

cap = cv2.VideoCapture("C:/Users/furka/Desktop/computer-vision/image-processing-projects/face-detection/video3.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.2) # optimize the parameter

mpDraw = mp.solutions.drawing_utils

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    #print(results.detections)

    if results.detections: # if not none
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box # bounding box corr

            h, w, _ = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h) # draw the box, x, y, w, h
            cv2.rectangle(img, bbox, (0,255,255),2)


    cv2.imshow("img", img)
    cv2.waitKey(1)
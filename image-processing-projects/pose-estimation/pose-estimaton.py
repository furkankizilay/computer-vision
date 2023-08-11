# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/

import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose # create mpPose object
pose = mpPose.Pose() # provide detection(false) tracking(true), default false
mpDraw = mp.solutions.drawing_utils # create mpDraw for vis

cap = cv2.VideoCapture("C:/Users/furka/Desktop/computer-vision/image-processing-projects/pose-estimation/video3.mp4")

pTime = 0

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks: # is not none
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)

            if id == 4:
                cv2.circle(img, (cx, cy), 5, (255,0,0),cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "FPS: "+ str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)


    cv2.imshow("img", img)
    cv2.waitKey(25) # set video speed



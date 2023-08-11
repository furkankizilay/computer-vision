# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

# frame -> per picture in video
# palm deteceter -> hand detection

import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0) # use default camera

mpHand = mp.solutions.hands # create mpHand object

mpDraw = mp.solutions.drawing_utils # create mpDraw for vis

pTime = 0
cTime = 0

hands = mpHand.Hands()  # static_image_mode is boolean, provide detection(false) tracking(true), default false 
                        # max_num_hands is number hands default 1
                        # min_detection_confidence, min_tra_confidence is a percentage value, it sets the ratio between detecting and tracking


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB) # process is called to use the module
    print(results.multi_hand_landmarks) # As long as multi_hand_landmarks does not show a hand image, it returns None, 
                                        # when it sees a hand image, it returns the coordinates of the points of the joints in the hand.

    if results.multi_hand_landmarks: # if not none
        for handLms in results.multi_hand_landmarks: # get the landmarks
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) # draw landmarks, mpHand.HAND_CONNECTIONS -> draw connection

            for id, lm in enumerate(handLms.landmark): # id is joint
                # print(id, lm)
                h, w, c = img.shape

                cx, cy = int(lm.x*w), int(lm.y*h) # get coordinate points

                # wrist
                if id == 4:
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
    

    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: "+str(int(fps)), (10, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0))


    cv2.imshow("img", img)
    cv2.waitKey(1)


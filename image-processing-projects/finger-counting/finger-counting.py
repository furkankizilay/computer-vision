import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3,640) # This line sets the width of the captured video frames to 640 pixels.
cap.set(4,480) # This line sets the height of the captured video frames to 480 pixels. 

mpHand = mp.solutions.hands # create mpHand object
hands = mpHand.Hands() # provide detection(false) tracking(true), default false
mpDraw = mp.solutions.drawing_utils # create mpDraw for vis

tipIds = [4, 8, 12, 16, 20] # finger three points

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB) # process is called to use the module
    # print(results.multi_hand_landmarks)

    lmList = []

    if results.multi_hand_landmarks: # if not none
        for handLms in results.multi_hand_landmarks: # get the landmarks
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) # draw landmarks, mpHand.HAND_CONNECTIONS -> draw connection

            for id, lm in enumerate(handLms.landmark): # id is joint
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # get coordinate points
                lmList.append([id, cx, cy]) # id, x corr, y corr

            """    
                # sign = 20
                if id == 20:
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
            
                # sign = 18
                if id == 18:
                    cv2.circle(img, (cx,cy), 9, (0,0,255), cv2.FILLED)
            """

    if len(lmList) != 0:
        fingers = []

        # thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1] :
            fingers.append(1)
        else:
            fingers.append(0)

        # four finger
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)

        totalF = fingers.count(1)
        #print(totalF)

        cv2.putText(img, str(totalF), (30,125), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)

    cv2.imshow("img", img)
    cv2.waitKey(1)
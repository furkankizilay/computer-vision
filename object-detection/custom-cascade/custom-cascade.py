"""
1- crate dataset: n, p
2- downland cascade program - https://amin-ahmadi.com/cascade-trainer-gui/
3- create cascade
4- write detection algorithm with using cascade
"""

import cv2
import os 

# images folder
path = "C:/Users/furka/Desktop/computer-vision/object-detection/custom-cascade/images"

# image shape
imgWidth = 180
imgHeight = 120

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180) # brightness

global countFolder
def saveDataFunction():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder +=1
    os.makedirs(path+str(countFolder))

saveDataFunction()
    
count = 0
countSave = 0

while True:
    success, cap = cap.read()

    if success:
        img =cv2.resize(img, (imgWidth, imgHeight))

        if count % 5 == 0:
            cv2.imwrite(path+str(countFolder)+"/"+str(countSave)+"_"+".png", img)
            countSave += 1
            print(countSave)
        count += 1

        cv2.imshow("image", img)

    if cv2.waitKey(0) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()



















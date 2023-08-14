"""
Object Detection using Haar feature-based cascading classifier is an effective method of object detection proposed by Paul Viola and Michael Jones in their 2001 article titled "Fast Object Detection Using Raised Simple Features Cascading".

The like features function is trained from many positive and negative images.

It is then used to detect objects in other images.

It is just like our convolutional core. Each feature is a single value obtained by subtracting the sum of the pixels below the white rectangle from the sum of the pixels below the black rectangle.

a) edge features
b) line features
c) four-rectangle features

https://github.com/opencv/opencv/tree/master/data/haarcascades
"""

import cv2
import matplotlib.pyplot as plt

einstein = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/face-detection/einstein.jpg",0)

plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

barca = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/face-detection/barcelona.jpg",0)

face_rect = face_cascade.detectMultiScale(barca, minNeighbors=6)

for (x,y,w,h) in face_rect:
    cv2.rectangle(barca, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barca, cmap = "gray"), plt.axis("off")

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)
            
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

plt.show()
cap.release()
cv2.destroyAllWindows()















# import the opencv and numpy libraries
import cv2
import numpy as np

# import the picture in black and white and draw the picture
img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/overview/overview.jpg", 0)
cv2.imshow("image", img)

# let's detect and visualize the edges on the image edge detection
edges = cv2.Canny(image=img, threshold1=200, threshold2=255)
cv2.imshow("edges", edges)

# import the haar cascade required for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(img)

# let's detect face and visualize the results
for (x,y,w,h) in face_rect:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255),10)
cv2.imshow("image with rectangle", img)

# Let's initialize HOG, call our human detection algorithm and set svm
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# apply our human detection algorithm to the picture and visualize it
(rects, weights) = hog.detectMultiScale(img, padding=(8,8), scale=1.05)
for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow("image with rectangle hog", img)

cv2.waitKey(0) & 0xFF == ord("q")
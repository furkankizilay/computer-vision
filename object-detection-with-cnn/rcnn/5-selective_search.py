"""
Selective Search for Object Detection

Selective Search is a method for segmenting an image into regions using a superpixel algorithm.

A superpixel can be defined as a group of pixels that share common characteristics, such as pixel intensity.

Selective Search hierarchically merges superpixels based on five primary similarity measures:

- Color similarity

- Texture similarity

- Size similarity

- Shape similarity

- Linear combination of the above similarities

Selective Search generates regions rather than class labels. These generated regions will be provided as input to the classifier.
"""
# This algorithm is an alternative to pyramid and sliding window methods.

import cv2
import random

img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection-with-cnn/rcnn/pyramid.jpg")
img = cv2.resize(img, dsize=(600,600))
cv2.imshow("img",img)

# initialize ss
ss =cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchQuality()

print("start")
rects = ss.process()

output = img.copy()

for (x,y,w,h) in rects[:50]:
    color = [random.randint(0,255) for j in range(0, 3)]
    cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

cv2.imshow("output", output)

k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()


























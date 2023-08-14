"""
Contour detection is the method that aims to connect all continuous points of the same color or density.

Contours are used for shape analysis and object detection and recognition
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/contour-detection/contour.jpg", 0)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("img1", img2)

# Ensure the image is in CV_8UC1 format
if img.dtype != np.uint8:
    img = img.astype(np.uint8)

contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# cv2.RETR_CCOMP means I want to find all internal and external contours
# cv2.CHAIN_APPROX_SIMPLE allows to compress horizontal, vertical, and diagonal sections, leaving only the endpoints

external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):

    # external
    if hierarch[0][i][3] == -1: # means external
        cv2.drawContours(external_contour, contours, i, 255, -1) # 255 color, -1 thickness

    else:
        cv2.drawContours(internal_contour, contours, i, 255, -1) # 255 color, -1 thickness

cv2.imshow("external", external_contour)
cv2.imshow("internal", internal_contour)


k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()


import cv2
import numpy as np

img = cv2.imread("images/Lenna_(test_image).png")
cv2.imshow("Original", img),

hor = np.hstack((img,img))
cv2.imshow("horizontal", hor)

ver = np.vstack((img,img))
cv2.imshow("vertical", ver)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
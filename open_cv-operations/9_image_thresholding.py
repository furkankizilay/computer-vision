import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", img)

# threshold
_, threh_img = cv2.threshold(img, thresh=60, maxval=255, type=cv2.THRESH_BINARY) # between 60 and 255 will make it white
cv2.imshow("threhed_image", threh_img)

# adaptive thresholding is used in order not to spoil the whole
# (maxval, adaptive method, threshold type, blocksize, _)
threh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
cv2.imshow("threhed_image2", threh_img2)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()



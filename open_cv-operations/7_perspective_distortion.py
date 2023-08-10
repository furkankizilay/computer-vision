import cv2
import numpy as np

img = cv2.imread("images/card.png")
cv2.imshow("original", img)

width = 400
height = 500

pts1 = np.float32([[230,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)

img_out = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow("original", img_out)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
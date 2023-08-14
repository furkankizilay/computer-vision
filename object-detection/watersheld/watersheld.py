"""
watersheld is the classical algorithm for segmentation, i.e. to separate different objects in an image.

Any grayscale image can be seen as a topographic surface, where high intensity denotes peaks and peaks and low intensity denotes valleys.

You start filling each isolated valley (local minimum) with different colored water (tags).

As the water rises, depending on the nearby peaks (gradients), the water from the different valleys will obviously start to merge with different colors.

To avoid this, you build barriers where the water meets. You continue the work of filling water and building barriers until all the peaks are submerged.

Then the obstacles you create will give you the segmentation result.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


coin = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/watersheld/coins.jpg")
cv2.imshow("coins", coin)

# reduce noise on coins, lpf blurring
coin_blur = cv2.medianBlur(coin, 13)
cv2.imshow("coins_blurr", coin_blur)

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
cv2.imshow("coins_gray", coin_gray)

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 75, 255, cv2.THRESH_BINARY)
cv2.imshow("coin_thresh", coin_thresh)

# contour
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if hierarchy[0][i][3] == -1 : # external
        cv2.drawContours(coin, contours, i, (0,255,0), 10)

cv2.imshow("coin", coin)


# watersheld
coin = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/watersheld/coins.jpg")
cv2.imshow("coins", coin)

# reduce noise on coins, lpf blurring
coin_blur = cv2.medianBlur(coin, 13)
cv2.imshow("coins_blurr", coin_blur)

# grayscale
coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
cv2.imshow("coins_gray", coin_gray)

# binary threshold
ret, coin_thresh = cv2.threshold(coin_gray, 65, 255, cv2.THRESH_BINARY)
cv2.imshow("coin_thresh", coin_thresh)

# opening -> erosion + dilation,
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
cv2.imshow("coin_thresh_opening", opening)

# distance between object
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
cv2.imshow("coin_thresh_distance", dist_transform)

# shrink image
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform),255,0)
cv2.imshow("sure_foreground", sure_foreground)

# enlarge the picture for the background
sure_background = cv2.dilate(opening, kernel, iterations = 1)
sure_foreground = np.uint8(sure_foreground)
unknown = cv2.subtract(sure_background,sure_foreground)
cv2.imshow("unknown", unknown)

# connection
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown == 255] = 0
#marker = np.uint8(marker)
#cv2.imshow("marker", marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

# watersheld
marker = cv2.watershed(coin,marker)
plt.figure(), plt.imshow(marker, cmap="gray"), plt.axis("off")

# contour
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):

    if hierarchy[0][i][3] == -1 : # external
        cv2.drawContours(coin, contours, i, (255,0,0), 2)

cv2.imshow("coin", coin)
plt.show()

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
























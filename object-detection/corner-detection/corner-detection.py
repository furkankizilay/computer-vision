"""
In computer vision we often need to find matching points between different frames of a picture or video. If it is known how the two images relate to each other, both images can be used to obtain information.

Matching points generally refer to easily recognizable features in the scene. We call these properties features.

These features must be uniquely recognizable.

Key features:
- Edges
- Corners

The vertices represent a point where the directions of these two sides change because they are the intersection of two sides.

We'll find this "variation" because the corners represent a variation in the gradient in images. We will look for variation in image density.

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/corner-detection/sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# harris corner detection
dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04) # blocksize is neighboring pixel, ksize is box size
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off")

# make corners more visible
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap="gray"), plt.axis("off")

# shi tomsai detection
img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/corner-detection/sudoku.jpg", 0)
img = np.float32(img)
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10) # 100 is corner count, 0.01 is quality level, 10 is min distance
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel() #  This line of code unpacks the flattened point into separate x and y variables for easier manipulation.
    cv2.circle(img, (x,y), 3, (125,125,125), cv2.FILLED)

plt.imshow(img), plt.axis("off"), plt.show()



















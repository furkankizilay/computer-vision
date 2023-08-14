# Edge detection is a method that tries to identify the points where the image brightness changes sharply.

import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the image
img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/edge-detection/london.jpg", 0)
cv2.imshow("img", img)
# plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# define the edges
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
cv2.imshow("edges", edges)
#plt.figure(), plt.imshow(edges, cmap="gray"), plt.axis("off")

med_val = np.median(img)
print(med_val)

# threshold optimization
low = int(max(0,1-0.33)*med_val)
high = int(min(255, (1+0.33))*med_val)
print(low), print(high)

# define the edges
edges = cv2.Canny(image=img, threshold1=low, threshold2=high)
cv2.imshow("edges_with_optimizaton", edges)
#plt.figure(), plt.imshow(edges, cmap="gray"), plt.axis("off")

# still have a lot of edges cause of that we should do blur

# blur
blurred_img = cv2.blur(img, ksize=(3,3))
cv2.imshow("blurred_img", blurred_img)
#plt.figure(), plt.imshow(blurred_img, cmap="gray"), plt.axis("off"), plt.show()

med_val_blurr = np.median(blurred_img)
print(med_val_blurr)

# threshold optimization
low_blurr = int(max(0,1-0.33)*med_val_blurr)
high_blurr = int(min(255, (1+0.33))*med_val_blurr)
print(low_blurr), print(high_blurr)

# define the edges
edges = cv2.Canny(image=blurred_img, threshold1=low_blurr, threshold2=high_blurr)
cv2.imshow("blurred_img_edges", edges)
#plt.figure(), plt.imshow(edges, cmap="gray"), plt.axis("off"), plt.show()

k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()


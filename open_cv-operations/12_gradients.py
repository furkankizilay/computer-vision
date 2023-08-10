"""
an image gradient is a directional change in intensity or color in an image.
used for edge detection.
"""

import cv2
import matplotlib.pyplot as plt


img = cv2.imread("images/sudoku.jpg", 0)
plt.figure()
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.title("original")
plt.show()

#  output depth 
# x gradient
sobelx = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5) # depth is precision of each pixel
plt.figure()
plt.imshow(sobelx, cmap = "gray")
plt.axis("off")
plt.title("sobel X")
plt.show()


# y gradient 
sobely = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5) # depth is precision of each pixel
plt.figure()
plt.imshow(sobely, cmap = "gray")
plt.axis("off")
plt.title("sobel Y")
plt.show()


# Laplacian  gradient
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure()
plt.imshow(laplacian, cmap = "gray")
plt.axis("off")
plt.title("Laplacian")
plt.show()




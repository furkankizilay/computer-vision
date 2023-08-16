"""
Sliding Window

A "sliding window" is a rectangular region of constant width and height that moves across an image.

For each of these windows, the window region is extracted, and an image classifier is applied within the window to determine whether there is an object of interest.

"""

import cv2
import matplotlib.pyplot as plt

def sliding_window(image, step, ws):

    for y in range(0, image.shape[0]-ws[1], step):
        for x in range(0, image.shape[1]-ws[0], step):
            yield(x, y, image[y:y+ws[1], x:x+ws[0]])

img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection-with-cnn/rcnn/husky.jpg")
#plt.imshow(img)
#print(img.shape)
#plt.show()
im = sliding_window(img, 5, (200,150))

"""for i, image in enumerate(im):
    print(i)
    if i == 1000:
        print(image[0], image[1])
        print(image)
        plt.imshow(image[2])
        plt.show()"""



















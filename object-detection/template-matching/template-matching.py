"""
Template mapping is used to search and find the location of a template image in a larger image.

It shifts the template image over the input image and compares the template and patch of the input image below the template image.

To move the template one pixel at a time by swiping.

At each location, a metric is calculated to represent how "good" or "bad" the match at that location is (or how similar the template is to that particular area of the source image).
"""

import cv2
import matplotlib.pyplot as plt

# template matching
img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/template-matching/cat.jpg", 0)
print(img.shape)
template = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/template-matching/cat_face.jpg", 0)
print(template.shape)
h, w = template.shape

# find the coordinate
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:

    method = eval(meth) # "cv2.TM_CCOEFF" -> cv2.TM_CCOEFF

    res =cv2.matchTemplate(img, template, method)
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] +h)

    cv2.rectangle(img, top_left, bottom_right, 255 ,2)

    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("matching result"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap="gray")
    plt.title("detecting result"), plt.axis("off")
    plt.suptitle(meth)
    plt.show()












"""
Pyramid Representation

It is a multi-scale representation of an image.

The utilization of an image pyramid enables us to detect objects in images at different scales.

At the base of the pyramid, there is the original image at its original dimensions (in terms of width and height). And at each subsequent layer, the image is resized (subsampled) and optionally smoothed (often through Gaussian blurring).

"""
import cv2
import matplotlib.pyplot as plt

def image_pyramid(image, scale = 1.5, minsSize=(224,224)):
    yield image

    while True:
        w = int(image.shape[1]/scale)
        image = cv2.resize(image, dsize=(w,w))

        if image.shape[0] < minsSize[1] or image.shape[1] < minsSize[0]:
            break

        yield image

"""img = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection-with-cnn/rcnn/husky.jpg")
im = image_pyramid(img, 1.5, (10,10))
for i, image in enumerate(im):
    print(i)
    if i == 8:
        plt.imshow(image)
        plt.show()"""












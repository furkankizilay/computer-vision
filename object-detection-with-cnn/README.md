# Object Detection with CNN and OpenCV

## RCNN

**1. Pyramid Representation**

It is a multi-scale representation of an image.

The utilization of an image pyramid enables us to detect objects in images at different scales.

At the base of the pyramid, there is the original image at its original dimensions (in terms of width and height). And at each subsequent layer, the image is resized (subsampled) and optionally smoothed (often through Gaussian blurring).

**2. Sliding Window**

A "sliding window" is a rectangular region of constant width and height that moves across an image.

For each of these windows, the window region is extracted, and an image classifier is applied within the window to determine whether there is an object of interest.

**3. Non-Maximum Suppression**

Non-maximum suppression disregards smaller overlapping bounding boxes and retains only the larger ones.

Intersection over Union (IoU) value is calculated between the boxes.

Boxes that fall below a certain threshold value are eliminated.

**4. Selective Search for Object Detection**

Selective Search is a method for segmenting an image into regions using a superpixel algorithm.

A superpixel can be defined as a group of pixels that share common characteristics, such as pixel intensity.

Selective Search hierarchically merges superpixels based on five primary similarity measures:

- Color similarity

- Texture similarity

- Size similarity

- Shape similarity

- Linear combination of the above similarities

Selective Search generates regions rather than class labels. These generated regions will be provided as input to the classifier.

## SSD

SSD divides the image using a grid instead of using a sliding window and assigns each grid cell the responsibility of detecting objects in that region of the image.

Object detection involves predicting the class and location of an object in that region.
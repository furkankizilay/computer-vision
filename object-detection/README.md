# Object Detection

Object detection in OpenCV is a computer vision technique that involves identifying and localizing specific objects within an image or a video stream. It goes beyond simple image classification by not only recognizing what's present in the image but also drawing bounding boxes around the detected objects to precisely locate them. OpenCV's object detection methods often utilize pre-trained models like Haar cascades or deep learning-based models such as YOLO (You Only Look Once) and Faster R-CNN to achieve accurate and efficient object localization and recognition in various applications like surveillance, robotics, and image analysis.

## 1. Edge Detection: 

Edge detection is a method that tries to identify the points where the image brightness changes sharply.

## 2. Corner Detection

In computer vision we often need to find matching points between different frames of a picture or video. If it is known how the two images relate to each other, both images can be used to obtain information.

Matching points generally refer to easily recognizable features in the scene. We call these properties features.

These features must be uniquely recognizable.

Key features:
- Edges
- Corners

The vertices represent a point where the directions of these two sides change because they are the intersection of two sides.

We'll find this "variation" because the corners represent a variation in the gradient in images. We will look for variation in image density.

## 3. Contour Detection

Contour detection is the method that aims to connect all continuous points of the same color or density.

Contours are used for shape analysis and object detection and recognition.

## 4. Object Detection with Color

We will learn how to detect objects in certain colors with the contour finding method.

Contours can simply be described as a curve connecting continuous points of the same color or density.

Contours are a useful tool for shape analysis and object detection and recognition.

## 5. Template Matching

Template matching is used to search and find the location of a template image in a larger image.

It shifts the template image over the input image and compares the template and patch of the input image below the template image.

To move the template one pixel at a time by swiping.

At each location, a metric is calculated to represent how "good" or "bad" the match at that location is (or how similar the template is to that particular area of the source image).

## 6. Feature Matching

Point feature mapping in image processing is an effective method for detecting a specified target in a complex scene.

This method detects single structures rather than multiple objects.

For example, using this method, the person can recognize a particular person on messy image, but not any other person.

The Brute-Force matcher matches the identifier of a feature in one image with all features of another image and returns a match by distance.

It is slow as it checks for match on all properties.

In scale-invariant feature transformation, key points are first extracted from a set of reference images and stored.

An object is searched in a new image by individually comparing each feature in the new image with this stored data and finding candidate matching features based on the Euclidean distance of the feature vectors.

## 7. Watersheld

Watersheld is the classical algorithm for segmentation, i.e. to separate different objects in an image.

Any grayscale image can be seen as a topographic surface, where high intensity denotes peaks and peaks and low intensity denotes valleys.

You start filling each isolated valley (local minimum) with different colored water (tags).

As the water rises, depending on the nearby peaks (gradients), the water from the different valleys will obviously start to merge with different colors.

To avoid this, you build barriers where the water meets. You continue the work of filling water and building barriers until all the peaks are submerged.

Then the obstacles you create will give you the segmentation result.

## 8. Face Detection

Object Detection using Haar feature-based cascading classifier is an effective method of object detection proposed by Paul Viola and Michael Jones in their 2001 article titled "Fast Object Detection Using Raised Simple Features Cascading".

The like features function is trained from many positive and negative images.

It is then used to detect objects in other images.

It is just like our convolutional core. Each feature is a single value obtained by subtracting the sum of the pixels below the white rectangle from the sum of the pixels below the black rectangle.

a) edge features

b) line features

c) four-rectangle features

https://github.com/opencv/opencv/tree/master/data/haarcascades

## 9. Cat Face Detection

https://github.com/opencv/opencv/tree/master/data/haarcascades

## 10. Custom Cascade

1. crate dataset: n, p

2. downland cascade program - https://amin-ahmadi.com/cascade-trainer-gui/

3. create cascade

4. write detection algorithm with using cascade

## 11. Pedestrian Detection
```python
# hog identifier 
hog = cv2.HOGDescriptor()
# add SVM to identifier
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

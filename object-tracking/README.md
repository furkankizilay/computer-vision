# Object Tracking

Object tracking in OpenCV refers to the process of monitoring and following the movement of a specific object within a sequence of frames in a video. It involves identifying the object in the initial frame and then continuously updating its position in subsequent frames as it moves. OpenCV provides various algorithms and techniques for object tracking, such as the MeanShift, CamShift, and KLT (KLT Tracker) methods. These algorithms use various strategies to track objects based on their appearance or motion characteristics, making it useful in applications like video surveillance, autonomous vehicles, and activity analysis.

In object detection, we identify an object in an image, draw a bounding box around it, and classify the object. 

Object tracking, on the other hand, is a discipline within computer vision that aims to follow moving objects across a sequence of video frames. These objects can include humans, animals, vehicles, or other interesting items like a soccer ball in a match.

Object tracking has numerous practical applications such as surveillance, medical imaging, traffic flow analysis, unmanned vehicles, people counting, audience flow analysis, and human-computer interaction.

Technically, object tracking starts with object detection, which involves identifying objects in an image and assigning bounding boxes to them.

An object tracking algorithm assigns an identity to each identified object in the image and then attempts to track this identity across subsequent frames, determining the new position of the same object as it moves.

**Re-identification:** Matching the same object across consecutive frames, as objects can move unpredictably in and out of the frame and need to be linked back to previously seen instances in the video.

**Occlusion:** Objects can become partially or fully occluded in certain frames due to the presence of other objects obstructing them, necessitating methods to handle such cases.

**Identity Ambiguity:** When two objects intersect, determining which is which becomes essential for accurate tracking.

**Motion Blur:** Objects can appear differently due to their own motion or camera movements, introducing challenges in maintaining consistent tracking.

**Viewpoint Variations:** Objects may appear drastically different from various viewpoints, demanding consistent identification across all perspectives.

**Scale Changes:** The scale of objects in a video can significantly alter, for instance, due to camera zooming.

**Lighting Conditions:** Changes in lighting throughout a video can profoundly impact how objects appear, making consistent detection more challenging.

## 1. Mean Shift Algorithm

Mean Shift, a clustering algorithm that iteratively assigns data points to clusters by shifting points towards the mode, which can be defined as the highest data point density.

Hence, it is also known as a mode-seeking algorithm.

## 2. Other Tracking Algorithm

Based on an online version of AdaBoost.
This classifier should be trained with positive and negative examples of the object at runtime.

The initial bounding box provided by the user (or another object detection algorithm) is taken as a positive example for the object, and many image patches outside the bounding box are considered as background.

When a new frame is given, the classifier is run on each pixel around the previous location, and the classifier's score is recorded.

The new position of the object is where the score is maximum.

https://motchallenge.net/data/MOT17/

https://arxiv.org/pdf/1603.00831.pdf

## 3. Multiple Object Tracking

Multiple object tracking in OpenCV involves simultaneously tracking the movement of multiple objects within a sequence of frames in a video. This process goes beyond single object tracking and requires algorithms that can handle the challenges of tracking multiple objects that may have similar appearances and interactions. OpenCV provides methods like the MultiTracker API, which combines various single-object tracking algorithms to track multiple objects efficiently. This technology finds applications in scenarios like crowd monitoring, object interaction analysis, and surveillance systems where tracking and understanding the movement of multiple objects are important.
"""
Non-Maximum Suppression

Non-maximum suppression disregards smaller overlapping bounding boxes and retains only the larger ones.
Intersection over Union (IoU) value is calculated between the boxes.
Boxes that fall below a certain threshold value are eliminated.
"""

# It helps reduce multiple detections for a single object.

import numpy as np
import cv2

def non_max_suppression(boxes, probs = None, overlapThresh=0.3):

    # Checks if the list of boxes is empty. If it is, an empty list is returned as there are no boxes to process.
    if len(boxes) == 0:
        return []
    
    # If the datatype of the boxes is integer, it converts them to float.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Extracts the coordinates of the boxes into separate arrays for x1, y1, x2, and y2.
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Calculates the area of each bounding box.
    area = (x2 - x1 + 1)*(y2 - y1 + 1)

    idxs = y2

    # probability
    if probs is not None:
        idxs = probs

    # sort index
    idxs = np.argsort(idxs)

    # Initializes an empty list pick to store the indexes of the chosen boxes after non-maximum suppression.
    pick = [] # chosen box

    # Enters a loop that continues until there are indexes left to process
    while len(idxs) > 0:

        # Picks the last index in the sorted list (idxs) and appends it to the pick list.
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # biggest x and y, calculates the intersection points between the current box and the remaining boxes in terms of x and y coordinates.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # w, h, computes the width and height of the overlapping region between boxes.
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)

        # overlap, calculates the overlap ratio (Intersection over Union) between the current box and the remaining boxes.
        overlap = (w*h)/area[idxs[:last]]

        # Deletes the indexes of boxes that have a high overlap (greater than the threshold) with the current box.
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Returns the final list of boxes that survived non-maximum suppression, converting their coordinates back to integers.
    return boxes[pick].astype("int")











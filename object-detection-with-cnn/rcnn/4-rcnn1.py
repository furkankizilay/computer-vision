from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

from sliding_window import sliding_window
from image_pyramid import image_pyramid
from non_max_supression import non_max_suppression

# initial paramater 
WIDTH = 600
HEIGHT = 600
PYR_SCALE = 1.5 # image pyramid scale
WIN_STEP = 16 # sliding step size
ROI_SIZE = (200,150)
INPUT_SIZE = (224, 224) # ResNet input size

print("Loading ResNet")
model = ResNet50(weights = "imagenet", include_top = True)

original = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection-with-cnn/rcnn/husky.jpg")
original = cv2.resize(original, dsize=(WIDTH, HEIGHT))
cv2.imshow("Husky", original)

(H, W) = original.shape[:2]

# image pyramid
pyramid = image_pyramid(original, PYR_SCALE, ROI_SIZE) # In each iteration, we will run a sliding window.

rois = []
locs = []

for image in pyramid:

    # In the image_pyramid function, we are using a scale to change the size of the image, and similarly, we need to apply this scale to the window sizes of the sliding window as well.
    scale = W/float(image.shape[1])
    
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)

        # I need to preprocess the image contained within the roiOrig cube to make it suitable for classification.
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        rois.append(roi)
        locs.append((x,y,x+w,y+h))

rois = np.array(rois, dtype="float32")

print("classification")

preds = model.predict(rois)

preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}
min_conf = 0.9 # 0.95, 0.8

for (i, p) in enumerate(preds):

    (_, label, prob) = p[0]

    if prob >= min_conf:

        box = locs[i]

        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

for label in labels.keys():

    clone = original.copy()

    for (box, prob) in labels[label]:
        (startX, starY, endX, endY) = box
        cv2.rectangle(clone, (startX, starY), (endX, endY), (0,255,0), 2)
    
    cv2.imshow("first", clone)

    clone = original.copy()

    # non-maxima

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])

    boxes = non_max_suppression(boxes, proba)

    for (startX, starY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, starY), (endX, endY), (0,255,0), 2)
        y = starY - 10 if starY - 10 > 10 else starY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    
    cv2.imshow("maxima",clone)

k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
    
    














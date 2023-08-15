"""
1. download dataset
2. convert image 2 video
3. eda -> gt
"""
"""
https://motchallenge.net/data/MOT17/
https://arxiv.org/pdf/1603.00831.pdf
"""

import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

pathIn = r'C:/Users/furka/Desktop/computer-vision/object-tracking/other-tracking-algorithm/img1'
pathOut = "C:/Users/furka/Desktop/computer-vision/object-tracking/other-tracking-algorithm/MOT17-13-SDP.mp4"

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]

#img = cv2.imread(pathIn + "\\" + files[44])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(img), plt.show()

fps = 25
size = (1920,1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

for i in files :
    print(i)
    filename = pathIn + "\\" + i
    img = cv2.imread(filename)
    out.write(img)

out.release()














"""
Point feature mapping in image processing is an effective method for detecting a specified target in a complex scene.

This method detects single structures rather than multiple objects.

For example, using this method, the person can recognize a particular person on messy image, but not any other person.

The Brute-Force matcher matches the identifier of a feature in one image with all features of another image and returns a match by distance.

It is slow as it checks for match on all properties.

In scale-invariant feature transformation, key points are first extracted from a set of reference images and stored.

An object is searched in a new image by individually comparing each feature in the new image with this stored data and finding candidate matching features based on the Euclidean distance of the feature vectors.
"""

import cv2
import matplotlib.pyplot as plt

chos = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/feature-matching/chocolates.jpg", 0)
cv2.imshow("chos", chos)

cho = cv2.imread("C:/Users/furka/Desktop/computer-vision/object-detection/feature-matching/nestle.jpg", 0)
cv2.imshow("nestle",cho)

# orb identifier
# corner - edge
orb = cv2.ORB_create()

# detect the point in orb
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# bf matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# match the point
matches = bf.match(des1, des2)

# sort by according to distance
matches = sorted(matches, key = lambda x: x.distance)

# vis matching image
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
cv2.imshow("orb", img_match)

# sift identifier  pip instal opencv-contrib-python --user
# sift and org only match the features
sift = cv2.xfeatures2d.SIFT_create()

# detect the point in sift
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

#bf
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good_matching = []
for match1, match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good_matching.append([match1])

sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, good_matching, None, flags=2)
cv2.imshow("sift", sift_matches)


k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()



























# morphological operations such as erosion, expansion, opening, closing and morphological gradient.

# erosion: its basic idea is just like erosion, eroding the boundaries of the foreground object.
# dilation: the opposite of erosion, it increases the white area in the image.
# opening: erosion + dilation, is used to prevent noise.
# closing: the opposite of opening, dilation + erosion. used to cover small holes in foreground objects or small black dots on the object.
# morphological gradient: it is the difference between dilation and erosion of an image.

import cv2
import numpy as np

img = cv2.imread("images/datai_team.jpg", 0)
cv2.imshow("image",img)

# erosion
kernel = np.ones((5,5), dtype=np.uint8)
result = cv2.erode(img, kernel, iterations=1)
cv2.imshow("erosion",result)

# dilation
result2 = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("dilation",result2)

# white noise
whiteNoise = np.random.randint(0, 2, size=img.shape[:2]) # 0-1 random integer
whiteNoise = whiteNoise*255

noise_img = whiteNoise + img
cv2.imshow("white noise image",noise_img.astype(np.float32))

# opening
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)
cv2.imshow("opening",opening)

# black noise
balckNoise = np.random.randint(0, 2, size=img.shape[:2]) # 0-1 random integer
balckNoise = balckNoise*-255

black_noise_img = balckNoise + img
black_noise_img[black_noise_img <= -245] = 0
cv2.imshow("black noise image",black_noise_img.astype(np.float32))

# closing
closing = cv2.morphologyEx(black_noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)

# morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("gradient", gradient)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
import cv2
import numpy as np

# create images
img = np.zeros((512,512,3), np.uint8) # black image
print(img.shape)

cv2.imshow("black", img)

# line
cv2.line(img, (0,0), (512,512),(0,255,0), 3) # (images, start point, end point, color, thickness)
cv2.imshow("black with line", img)

# rectangle
cv2.rectangle(img, (0,0),(255,255), (255,0,0), cv2.FILLED) # # (images, start point, end point, color)
cv2.imshow("black with rectangle", img)

# circle
cv2.circle(img, (300,300), 50, (0,0,255), cv2.FILLED) # (images, start point, r, color)
cv2.imshow("black with circle", img)

# text
cv2.putText(img, "image", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255)) # (images, name, start point, font, thickness, color)
cv2.imshow("black with text", img)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()



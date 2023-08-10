import cv2
import matplotlib.pyplot as plt

# blending

img1 = cv2.imread("images/img1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # opencv's default paramater is bgr convert it rgb
img2 = cv2.imread("images/img2.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

cv2.imshow("image1",img1)
cv2.imshow("image2",img2)

# for blending we need same shape 
print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (600,600))
img2 = cv2.resize(img2, (600,600))

print(img1.shape)
print(img2.shape)

cv2.imshow("image1",img1)
cv2.imshow("image2",img2)

# blending picture = alpha*img1 + beta*img2
blended = cv2.addWeighted(src1=img1, alpha=0.3, src2=img2, beta=0.7, gamma=0)
cv2.imshow("blanded image",blended)


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('images/blended', blended) # -> save the image
    cv2.destroyAllWindows() # -> close the all windows

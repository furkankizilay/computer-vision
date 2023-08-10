import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/overview.jpg", 0)
cv2.imshow("picture", img)

print(img.shape) # (568, 860)

resized_img = cv2.resize(img,(int(img.shape[1]*(4/5)), (int(img.shape[0]*(4/5)))))
cv2.imshow("resized_picture", img)

img = cv2.putText(img,"dog",(350,200),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
cv2.imshow("imgWithText", img)

_, thresh_img = cv2.threshold(img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
cv2.imshow("thresh_img", thresh_img)

adap_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,8)
cv2.imshow("adap_thresh", adap_thresh)

gauss_img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
cv2.imshow("gauss_img", gauss_img)

lablacian_img = cv2.Laplacian(img,ddepth=cv2.CV_64F)
cv2.imshow("lablacian_img", lablacian_img)

hist_img = cv2.calcHist([img],channels=[0], mask=None, histSize=[256],ranges=[0,256])
plt.figure()
plt.plot(hist_img)
plt.show()


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()

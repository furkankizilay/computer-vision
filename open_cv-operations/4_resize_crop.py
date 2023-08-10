import cv2

img = cv2.imread("images/Lenna_(test_image).png")
print("image shape: ", img.shape)
cv2.imshow("origin", img)


# resize the image
img_resize = cv2.resize(img, (800,800))
print("resized img shape: ", img_resize.shape)
cv2.imshow("resized", img_resize)

# crop the image
img_crop = img[:200, 0:300] # -> height width
cv2.imshow("croped image: ", img_crop)
cv2.waitKey()
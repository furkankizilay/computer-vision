import cv2

# import the image
img = cv2.imread("images/messi5.jpg", 0) # 0's mean grayscale

# visualization
cv2.imshow("first image", img)
k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('images/messi_gray.png', img) # -> save the image
    cv2.destroyAllWindows() # -> close the all windows

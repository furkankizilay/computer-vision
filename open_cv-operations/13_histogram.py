"""
an image histogram is a type of histogram that functions as a graphical representation of the tonal distribution in a digital image.
contains the number of pixels for each tonal value.
by looking at the histogram for a particular image, the tonal distribution can be understood.

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("red blue", img_vis.astype(np.float32))

print(img_vis.shape)

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256]) 
print(img_hist.shape)
plt.figure()
plt.plot(img_hist)
plt.show()

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    img_hist = cv2.calcHist([img], channels = [i], mask = None, histSize = [256], ranges = [0,256]) 
    plt.plot(img_hist, color=c)

plt.show()

# mask
golden_gate = cv2.imread("images/goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis), plt.show()
#cv2.imshow("goldan gate", golden_gate_vis)

print(golden_gate_vis.shape)

mask = np.zeros(golden_gate.shape[:2], np.uint8)
plt.figure(), plt.imshow(mask, cmap="gray"), plt.show()

mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap="gray"), plt.show()

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask=mask)
plt.figure(), plt.imshow(masked_img_vis, cmap="gray"), plt.show()

masked_img = cv2.bitwise_and(golden_gate, golden_gate_vis, mask=mask)
masked_img_hist = cv2.calcHist([golden_gate], channels = [0], mask = mask, histSize = [256], ranges = [0,256]) 
plt.figure(), plt.plot(masked_img_hist), plt.show()


# histogram equalization (increase the contrast)
img = cv2.imread("images/hist_equ.jpg", 0)
plt. figure(), plt.imshow(img, cmap="gray"), plt.show()

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256]) 
plt.figure(), plt.plot(img_hist), plt.show()

# expanded the narrow region that was stuck between 120-200 to the range of 0-255.

eq_hist = cv2.equalizeHist(img)
plt. figure(), plt.imshow(eq_hist, cmap="gray"), plt.show()

eq_img_hist = cv2.calcHist([eq_hist], channels = [0], mask = None, histSize = [256], ranges = [0,256]) 
plt.figure(), plt.plot(eq_img_hist), plt.show()


k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()


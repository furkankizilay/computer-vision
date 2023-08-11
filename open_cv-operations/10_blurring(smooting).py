"""
image blur is achieved by applying a low-pass filter to the image.

useful for removing noise. removes high frequency content (noise, edges) from the image.

opencv provides three main blur techniques.
- average blur
- gaussian blur
- median blur

average blur 

it is done by wrapping an image with a normalized box filter.
takes the average of all pixels under the kernel area and replaces this average with the central element.

gaussian blur

this method uses gaussian kernel instead of box filter.
the width and height of the core, which must be positive and unique, are specified.
sigmaX and sigmaY, we must specify the standard deviation in the X and Y directions.

median blur 

takes the median of all pixels under the kernel area and the central element is replaced by that median value.
Its is very wary of salt and pepper noise.

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# blurring (reduces detail and prevents noise)
img = cv2.imread("images/NYC.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("new_york", img)
"""
# average blurr
dts2 = cv2.blur(img, ksize=(3,3))
cv2.imshow("average blurr",dts2)

# gauss blurr
gb = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7) # if sigmaY do not assign the sigmaY = sigmaX
cv2.imshow("gauss blurr",dts2)

# median blurr
mb = cv2.medianBlur(img, ksize=3)
cv2.imshow("median blurr",dts2)
"""
# create noise
def gaussianNoise(image):
    
    row,col,ch= image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    
    return noisy

def saltPepperNoise(image):

    row,col,ch = image.shape
    s_vs_p = 0.5 # black white points ratio

    amount = 0.004
    noisy = np.copy(image)

    # salt
    num_salt = np.ceil(amount*image.size*s_vs_p) # number of white noisy count
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape] # coords white noisy
    noisy[coords] = 1 # add

    # papper 
    num_papper = np.ceil(amount*image.size*1-s_vs_p) # number of black noisy count
    coords = [np.random.randint(0, i-1, int(num_papper)) for i in image.shape] # coords black noisy
    noisy[coords] = 0 # add

    return noisy

# normlized the image
img2 = cv2.imread("images/NYC.jpg")/255
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
cv2.imshow("new_york2", img2)
print(img2)

gaussiaNoisyImage = gaussianNoise(img2)
cv2.imshow("gaussionNoisyImage", gaussiaNoisyImage)

# gauss blurr
gb = cv2.GaussianBlur(gaussiaNoisyImage, ksize=(3,3), sigmaX=7) # if sigmaY do not assign the sigmaY = sigmaX
cv2.imshow("with gauss blurr",gb)

# median blurr
spImage = saltPepperNoise(img2)
cv2.imshow("SP median blurr", spImage)

# median blurr
mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize=3)
cv2.imshow("with median blurr",mb2)

k = cv2.waitKey(0) & 0xFF # -> wait the key

if k == 27: # wsc -> mean space
    cv2.destroyAllWindows()
















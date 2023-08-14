import cv2
import pickle # saves the marked image

width = 27
height = 15

try:
    with open("CarParkPos", "rb") as f:
        posList = pickle.load(f)
except:
    posList = []

def mouseClick(events, x, y, flags, params): # will leave a rectangle where pressed with the mouse

    if events == cv2.EVENT_LBUTTONDOWN: # press left click
        posList.append((x, y))

    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            # This condition checks if both the x and y coordinates of the mouse click are within the defined rectangular region.
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open("CarParkPos","wb") as f:
        pickle.dump(posList, f)

while True:

    img = cv2.imread("C:/Users/furka/Desktop/computer-vision/image-processing-projects/parking-space-counter/first_frame.png")

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255,0,0), 2) # draw the rectangle

    # print("postlist: ", posList)

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", mouseClick)
    cv2.waitKey(1)
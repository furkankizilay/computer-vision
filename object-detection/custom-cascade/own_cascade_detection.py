import cv2

path = "C:/Users/furka/Desktop/computer-vision/object-detection/custom-cascade/cascade.xml"
objectName = "pen tip"
frameWidth = 280
frameHeight = 360
color = (255,0,0)

cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

def empty(a):pass

# trackbar
cv2.namedWindow("conclusion")
cv2.resizeWindow("conclusion", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale","conclusion",400,1000,empty)
cv2.createTrackbar("Neighbor","conclusion",4,50,empty)

# cascade classifier
cascade = cv2.CascadeClassifier(path)

while True:
    
    # read img
    success, img = cap.read()
    
    if success:
        # convert bgr2gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detection parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Scale","conclusion")/1000)
        neighbor = cv2.getTrackbarPos("Neighbor","conclusion")
        # detectiom
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
    
        for (x,y,w,h) in rects:
            
            cv2.rectangle(img, (x,y),(x+w,y+h), color, 3)
            cv2.putText(img, objectName, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            
        cv2.imshow("conclusion", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

import cv2
import time

# get video name
video_name = "images/MOT17-04-DPM.mp4"

# capture, cap
cap = cv2.VideoCapture(video_name)

print("weight: ", cap.get(3))
print("height: ", cap.get(4))

if cap.isOpened() == False:
    print("warning")

while True:

    ret, frame = cap.read() # -> frame is images in video, ret is transaction success

    if ret == True:
        time.sleep(0.01) # to reduce video speed
        cv2.imshow("video",frame)

    else: break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() # -> stop capture
cv2.destroyAllWindows()
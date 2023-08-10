import cv2

# capture

cap = cv2.VideoCapture(0) # zero = default camera

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

# record video

writer = cv2.VideoWriter("images/first-video.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 20, (width,height))

# VideoWriter_fourcc -> 4-character codec code used to compress videos (*"DIVX" for windows) 20 is fps

while True:
    ret, frame = cap.read()
    cv2.imshow("video", frame)

    # save
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q") : break

cap.release() # stop capture
writer.release()
cv2.destroyAllWindows()
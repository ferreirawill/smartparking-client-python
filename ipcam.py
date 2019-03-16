import cv2
import time
import imutils
#cap = cv2.VideoCapture('http://admin:admin@192.168.0.29:80/cgi-bin/mjpg/video.cgi?&subtype=1')
#cap = cv2.VideoCapture('rtsp://admin:admin@192.168.0.29:554/cam/realmonitor?channel=1&subtype=0')
#cap = cv2.VideoCapture("http://admin:admin@192.168.0.29/cgi-bin/mjpg/video.cgi?&subtype=1")
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.0.29:554/cam/realmonitor?channel=1&subtype=0')

while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=1200)
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

from __future__ import print_function
from identifica.identifica import detector
from reconhece.reconhece import bbps
from imutils import paths
import numpy as np
import pickle
import argparse
import imutils
import cv2
import sys

#cap = cv2.VideoCapture("http://admin:admin@192.168.0.29/cgi-bin/mjpg/video.cgi?&subtype=1")
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.0.7:554/cam/realmonitor?channel=1&subtype=0')


charModel_dire =  "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_char.cpickle" 
numModel_dire = "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_digit.cpickle" 

charModel = pickle.loads(open(charModel_dire,"rb").read(), encoding= 'latin1')
numModel = pickle.loads(open(numModel_dire,"rb").read(), encoding= 'latin1')

blockSizes = ((5,5), (5,10), (10,5), (10,10))
desc = bbps(targetsize=(30,15), blocksizes=blockSizes)

while True:
    _, image = cap.read()
 

    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)
    lpd = detector(image,numchar=7)
    plates = lpd.detecta()
    for (lpBox, chars) in plates:
        # restructure lpBox
        lpBox = np.array(lpBox).reshape((-1, 1, 2)).astype(np.int32)
        text = ""

        for (i, char) in enumerate(chars):
            # preprocess the character and describe it
            char = detector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)

            # if this is the first 3 characters, then use the character classifier
            if i < 3:
                prediction = charModel.predict(features)[0]

            # otherwise, use the digit classifier
            else:
                prediction = numModel.predict(features)[0]

            # update the text of recognized characters
            text += prediction.upper().decode('utf-8')
        
        
        M = cv2.moments(lpBox)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the license plate region and license plate text on the image
        cv2.drawContours(image, [lpBox], -1, (0, 255, 0), 2)
        cv2.putText(image, text, (cX - (cX // 5), cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2)

    #cv2.imshow('Camera', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

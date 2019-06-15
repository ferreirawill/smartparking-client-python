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
import requests 
import json 
import time
from multiprocessing import Process

headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache",
    }
url = "http://127.0.0.1:8000/api/entrada/"
img = 'jhafqewuoh93r8h12093uawhflkwerh930hwewqehr'



#cap = cv2.VideoCapture("http://admin:admin@192.168.0.29/cgi-bin/mjpg/video.cgi?&subtype=1")
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.0.7:554/cam/realmonitor?channel=1&subtype=0')

charModel_dire =  "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_char.cpickle" 
numModel_dire = "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_digit.cpickle" 

charModel = pickle.loads(open(charModel_dire,"rb").read(), encoding= 'latin1')
numModel = pickle.loads(open(numModel_dire,"rb").read(), encoding= 'latin1')

blockSizes = ((5,5), (5,10), (10,5), (10,10))
desc = bbps(targetsize=(30,15), blocksizes=blockSizes)

def recognizement(image):
    text = ""

    lpd = detector(image,numchar=7)
    plates = lpd.detecta()
    for (lpBox, chars) in plates:


        for (i, char) in enumerate(chars):
            char = detector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)

            if i < 3:
                prediction = charModel.predict(features)[0]
            else:
                prediction = numModel.predict(features)[0]
            text += prediction.upper().decode('utf-8')

        #justplate = lpd.PlateImage(lpBox, text)
        print(text)


    

count=0
frame=0
while True:
    _, image = cap.read()
    frame=frame + 1
    count=count + 1
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)
    if count > 10:
        count=0
        p = Process(target=recognizement,args=(image,))
        p.start()
        p.join()

    cv2.imshow('Camera', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

"""
        elif text[6:8] == '09':
            text = 'WILL2609'
            justplate = lpd.PlateImage(lpBox, text)
            payload = {'placa': text,
                'img': justplate}
            payload = json.dumps(payload)
            response = requests.request("POST", url, data=payload, headers=headers) 
            print(response.text)
            
"""

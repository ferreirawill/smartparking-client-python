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


command = argparse.ArgumentParser()
command.add_argument("-i","--image", help="Imagens para deteccao")

dictio = vars(command.parse_args())
print(dictio)

charModel_dire =  "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_char.cpickle" 
numModel_dire = "/home/william/PycharmProjects/lpr_course/improving_classifier/output/adv_digit.cpickle" 

charModel = pickle.loads(open(charModel_dire,"rb").read(), encoding= 'latin1')
numModel = pickle.loads(open(numModel_dire,"rb").read(), encoding= 'latin1')

blockSizes = ((5,5), (5,10), (10,5), (10,10))
desc = bbps(targetsize=(30,15), blocksizes=blockSizes)


for imagePath in sorted(list(paths.list_images(dictio["image"]))):
    image = cv2.imread(imagePath)


    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = detector(image,numchar=7)
    plates = lpd.detecta()

    for(lpBox,chars) in plates:

        lpBox = np.array(lpBox).reshape((-1,1,2)).astype(np.int32)

        text = ""

        for(i, char) in enumerate(chars):

            char = detector.preprocessChar(char)
            if char is None:
                continue
            features = desc.describe(char).reshape(1, -1)

            if i < 3:
                prediction = charModel.predict(features)[0]
            else:
                prediction = numModel.predict(features)[0]

            text += prediction.upper().decode('utf-8')
            


            #cv2.imshow("Character {}".format(i + 1), char)


    cv2.imshow("Imagem",image)
    print("For image: {} , the plate is: {}".format((imagePath[imagePath.rfind("/") + 1:]),text))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#'''

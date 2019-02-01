from __future__ import print_function
from identifica.identifica import detector
from reconhece.reconhece import bbps
from imutils import paths
import numpy as np
import pickle
import argparse
import imutils
import cv2

command = argparse.ArgumentParser()
command.add_argument("-i","--image", required= True, help="Imagens para detecção")
command.add_argument("-c","--char", required= True, help="char para para detecção")
command.add_argument("-n","--num", required= True, help="num para para detecção")

dictio = vars(command.parse_args())

charModel = pickle.loads(open(dictio["char"],"rb").read())
numModel = pickle.loads(open(dictio["num"],"rb").read())

blockSizes = ((5,5), (5,10), (10,5), (10,10))
desc = bbps(targetsize=(30,15), blocksizes=blockSizes)

for imagePath in sorted(list(paths.list_images(dictio["image"]))):

    print(imagePath[imagePath.rfind("/") + 1:])
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
            features = desc.describe(char).reshape(1,-1)

            if i<3:
                prediction = charModel.predict(features)[0]
            else:
                prediction = numModel.predict(features)[0]

            text +=prediction.upper()

            if len(chars) >0:
                M = cv2.moments(lpBox)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            cv2.drawContours(image,lpBox,-1,(0,255,0),2)
            cv2.putText(image,text,(cX - (cX // 5), cY -30), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)


            #cv2.imshow("Character {}".format(i + 1), char)


    cv2.imshow("Imagem",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

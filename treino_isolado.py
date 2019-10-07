from __future__ import print_function
from identifica.identifica import detector
import traceback
import numpy as np
import random
import os
from imutils import paths
import argparse
import cv2
import imutils

command = argparse.ArgumentParser()
command.add_argument("-i","--images",required=True,help="Caminho das imagens")
command.add_argument("-e","--examples",required=True,help="Caminho das saidas")
dictio = vars(command.parse_args())

imagePaths = list(paths.list_images(dictio["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:int(len(imagePaths) * 0.5)]
counts = {}

for imagePath in imagePaths:
    print("[EXAMINANDO] {}".format(imagePath))

    try:
        image = cv2.imread(imagePath)
        if image.shape[1] >640:
            image = imutils.resize(image, width=640)

        lpd = detector(image,numchar= 7)
        plates = lpd.detecta()

        for (lpbox, chars) in plates:

            lpbox = np.array(lpbox).reshape((-1,1,2)).astype(np.int32)

            plate = image.copy()
            cv2.drawContours(plate, [lpbox],-1,(0,255,0),2)
            cv2.imshow("Placa",plate)

            for char in chars:

                cv2.imshow("char",char)
                key = cv2.waitKey(0)

                if key == ord(","):
                    print("[IGNORING] {}".format(imagePath))
                    continue

                key = chr(key).upper()
                dirpath = "{}/{}".format(dictio["examples"],key)

                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)

                count = counts.get(key,1)
                path = "{}/{}.png".format(dirpath,str(count).zfill(5))
                cv2.imwrite(path,char)

                counts[key] = count + 1





    except KeyboardInterrupt:
        break

    except:
        print(traceback.format_exc())
        print("[ERROR] {}".format(imagePath))

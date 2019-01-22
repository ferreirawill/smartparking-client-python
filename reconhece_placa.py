from __future__ import print_function
from identifica.identifica import detector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

command = argparse.ArgumentParser()
command.add_argument("-i","--image", required= True, help="Imagens para detecção")
dictio = vars(command.parse_args())


for imagePath in sorted(list(paths.list_images(dictio["image"]))):
    image = cv2.imread(imagePath)
    print(imagePath)

    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = detector(image)
    plates = lpd.detecta()

    for(lpBox,chars) in plates:
        for(i, char) in enumerate(chars):
            cv2.imshow("Character {}".format(i + 1),char)


    cv2.imshow("Imagem",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

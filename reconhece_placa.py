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
    plates = lpd.detectaplacas()

    for (i,(lp, lpBox)) in enumerate(plates):
        lpBox = np.array(lpBox).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(image, [lpBox], -1,(0,255,0),2)

        candidates = np.dstack([lp.candidates] * 3)
        thresh = np.dstack([lp.thresh] * 3)
        output = np.vstack([lp.plate, thresh, candidates])
        cv2.imshow("Plate e candidates. # {}".format(i + 1), output)

    cv2.imshow("Imagem",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
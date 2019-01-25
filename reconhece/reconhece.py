import numpy as np
import cv2

class bbps:
    def __init__(self,targetsize=(30,15),blocksizes=((5,5),)):
        self.targetsize = targetsize
        self.blocksizes = blocksizes

    def describe(self,image):
        image = cv2.resize(image,(self.targetsize[1],self.targetsize[0]))
        features = []

        for (blockW,blockH) in self.blocksizes:
            for y in range(0,image.shape[0],blockH):
                for x in range(0,image.shape[1],blockW):
                    roi = image[y:y + blockH, x:x +blockW]
                    total = cv2.countNonZero(roi) / float(roi.shape[0]*roi.shape[1])

                    features.append(total)


        return np.array(features)

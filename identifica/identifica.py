import numpy as np
import cv2
import imutils

class detector:
    def __init__(self,image,largmin=60,altmin=20):
        self.image = image
        self.largmin = largmin
        self.altmin = altmin

    def detecta(self):
        return self.detectaplacas


    def detectaplacas(self):
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        regions=[]

        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2. MORPH_BLACKHAT, rectKernel)

        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        #cv2.imshow("gray", gray)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("thresh", light)
        gradX = cv2.Sobel(blackhat, ddepth= cv2.CV_32F
                    if imutils.is_cv2() else cv2.CV_32F,dx=1,dy=0,ksize =-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        gradX = cv2.GaussianBlur(gradX,(5,5),0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        cv2.imshow("thresh", thresh)
        cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (w,h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            rect = cv2.minAreaRect(c)

            box = np.int0(cv2.boxPoints(rect))

            if(aspectRatio > 3 and aspectRatio < 6) and h > self.altmin and w > self.largmin:
                regions.append(box)

        return regions







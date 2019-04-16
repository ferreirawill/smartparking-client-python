from collections import namedtuple  # Cria uma tupla com um nome definido
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective # Classse para fazer a transformação de perspectiva das imagens
import numpy as np #Modulo de algebra linear do Python
import cv2 #Modulo de processamento de imagem em python
import imutils #Modulo para manipulacao de imagem em python


# Define uma tupla nomeada "placa_regiao" com parametros relativos a placa
# Semelhante a um struct c/c++, define um tipo de dados
licence_plate = namedtuple("placa_regiao",["success", "plate", "thresh", "candidates"])
#success = true or false se região detectada ou não
#plate = imagem da placa detectada
#thresh = threshold da imagem
#candidates = lista de caracteres da placa

# classe responsavel por detectar, extrair e reconhecer os caracteres da placa
class detector:
    # Construtor da classe
    def __init__(self,image,largmin=60,altmin=20,numchar=7,largminchar=40):
        self.image = image     # Recebe a imagem 
        self.largmin = largmin # Recebe a largura minima da placa
        self.altmin = altmin   # Recebe a altura minima da placa 
        self.numchar = numchar # Recebe o numero de caracteres da placa
        self.largminchar = largminchar # Recebe a largura minima de um caracter


    # Método principal para reconhecimento de placas
    def detecta(self):
        # Chama metodo para detectar placa na imagem
        regpl = self.detectaplacas() 
        
        # Faz um loop sobre toda regiao da placa
        for reg in regpl:
            # Chama metodo para detectar possiveis caracteres    
            lp = self.detectcharcandidates(reg)

            # Se for detectada uma placa
            if lp.success:
                # Chama metodo de corte de caracteres
                chars = self.scissor(lp)
                # retorna uma tupla a regiao e os caracteres
                yield(reg, chars)


        #return self.detectaplacas()

# Metodo responsavel por detectar a placa em uma imagem
    def detectaplacas(self):
        # Cria um kernel retangular que deslizara pela imagem ate encontrar uma placa
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        # Cria kernel quadrado que limpara os ruidos na imagem
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Cria lista com delimitação da placa
        regions = []
        #cv2.imshow("Imagem original",self.image)
        # Converte imagem para cinza 
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #cv2.MORPH_BLACKHAT revela regiões mais escuras na imagem
        #tudo que for preto será destacado
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        #cv2.imshow("BLACKHAT", blackhat)

        #cv2.MORPH_CLOSE retira ruidos
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        #cv2.imshow("gray", gray)
        # threshold binariza a imagem, pixel > 50 recebe valor 255
        # metodo cv2.threshold retorna uma lista com dois parametros: erro medio e imagem binarizada
        # light recebe binarizacao das partes escuras e claras da imagem
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
        #Aplica filtro de sobel para destacar o que não é preto no branco
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1,dy=0,ksize =-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        
        #cv2.imshow("CV_32 Normalizado", gradX) 
        
        # Desfoca imagem para tirar destaque de pequenos pixels deixado pelo sobel
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        #cv2.imshow("gradX", gradX)
        # Morph_close força a retirada de ruidos deixado pelo sobel
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        #cv2.imshow("morph_close", gradX)
        # OTSU threshold encontra o melhor valor entre os picos do histograma
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv2.imshow("thresh", thresh)
        # Operaçoes para de erosao e dilatacao para retirar blocos indesejados e expandir o quadrado da placa
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        #cv2.imshow("eroded and dilated", thresh)
        # Realiza um and na imagem thresh com ela mesma e usa como mascara o tresh_inv light
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        #cv2.imshow("finalizada", thresh) 
        
        # Encontra os contronos da imagem
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        for c in cnts:
            # Pega altura e largura do retangulo
            (w, h) = cv2.boundingRect(c)[2:]
            # calcula a relação entre altura e largura
            aspectRatio = w / float(h)
            

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            
            if (aspectRatio > 3 and aspectRatio < 6) and h > self.altmin and w > self.largmin:
                # Coloca as cordenadas da placa num array 
                regions.append(box)



        return regions



    def detectcharcandidates(self, region):
        # Redimensiona a imagem aproximando a região da placa
        plate = perspective.four_point_transform(self.image, region)
        #cv2.imshow("Transformação de prespectiva", imutils.resize(plate, width= 400))
        
        # Extrai o componente V do espaço de cores HSV 
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        #V = cv2.split(HSV)[2] # H = 0, S = 1, V =2
        #T = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #Se o pixel tiver uma intensidade maior que o threshold local o valor V sobressai
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        #thresh = cv2.erode(thresh, (2,2), iterations=1)
        #thresh = cv2.dilate(thresh, (2,2), iterations=1)
        #cv2.imshow("bitwise_not", thresh)

        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        #cv2.imshow("Thresh",thresh)
        
        # Tecnica "connected component labeling" para encontrar a forma da letra
        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])


                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                if keepAspectRatio and keepSolidity and keepHeight:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)


            charCandidates = segmentation.clear_border(charCandidates)
            cnts = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL,
             cv2.CHAIN_APPROX_SIMPLE)

            cnts = imutils.grab_contours(cnts)
            #cv2.imshow("Original candidates",charCandidates)
        if len(cnts) > self.numchar:
            (charCandidates, cnts) = self.prunecandidates(charCandidates, cnts)
            #cv2.imshow("Pruned Candidates",charCandidates)
        
        return licence_plate(success=len(cnts) == self.numchar, plate=plate, thresh=thresh, candidates=charCandidates)

            #thresh = cv2.bitwise_not(thresh, thresh, mask=charCandidates)
            #cv2.imshow("char threshold",thresh)

        

    def prunecandidates(self,charCandidates,cnts):
        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dims = []
        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dims.append(boxY + boxH)

        dims = np.array(dims)
        diffs = []
        selected = []

        for i in range(0, len(dims)):
            diffs.append(np.absolute(dims - dims[i]).sum())

        for i in np.argsort(diffs)[:self.numchar]:

            cv2.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
            selected.append(cnts[i])

        return (prunedCandidates, selected)


    def scissor(self,lp):
        cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL,
         cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        boxes = []
        chars = []

        for c in cnts:

            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.largminchar,self.largminchar - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)

            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        boxes = sorted(boxes, key=lambda b:b[0])

        for (startX, startY, endX, endY) in boxes:

            chars.append(lp.thresh[startY:endY, startX:endX])

        return chars


    def PlateImage(self, region, number):

        cutted = perspective.four_point_transform(self.image, region)
        cv2.imshow("PlateImage", imutils.resize(cutted, width= 400))
        pathimage='/home/william/PycharmProjects/smartparking_webservice/src/media_files/'+ number +".png"
        cv2.imwrite(pathimage,cutted)

        print(pathimage)


        return pathimage


    @staticmethod
    def preprocessChar(char):
        cnts = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            return None
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        char = char[y:y + h, x:x + w]

        return char
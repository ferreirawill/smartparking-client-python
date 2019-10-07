from __future__ import print_function
from reconhece.reconhece import bbps
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import pickle
import cv2
import imutils


command = argparse.ArgumentParser()
command.add_argument("-f","--fonts", required= True, help="Caminho das fontes")
command.add_argument("-c","--char",required=True,help="Caminho onde será salvo o treino de caracteres")
command.add_argument("-n","--num",required=True,help="Caminho onde será salvo o treino de numeros")
dictio = vars(command.parse_args())
print(dictio)

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

alphadata = []
numdata = []
alphalabel = []
numlabel = []

print("[INFO] Descrevendo exemplos de fontes...")
blocksizes = ((5,5),(5,10),(10,5),(10,10))
desc = bbps(targetsize=(30,15),blocksizes=blocksizes)

for fontpath in paths.list_images(dictio["fonts"]):
    font = cv2.imread(fontpath)
    font = cv2.cvtColor(font, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(font, 128, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c:(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1]))


    for (i,c) in enumerate(cnts):
        #print("Valor de i: {} | letra equivalente: {}".format(i,alphabet[i]))
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        features = desc.describe(roi)

        if i < 26:
            alphadata.append(features)
            alphalabel.append(alphabet[i])
            #print("ALPHALABEL: {}".format(alphalabel))
        elif i <36:
            numdata.append(features)
            numlabel.append(alphabet[i])
            #print("numlabel: {}".format(numlabel))


print("[INFO] Criando modelo de caracteres...")
charmodel = LinearSVC(C=1.0,random_state=42)
charmodel.fit(alphadata, alphalabel)

print("[INFO] Finalizando modelo de caracteres...")
f = open(dictio["char"],"wb")
f.write(pickle.dumps(charmodel))
f.close()

print("[INFO] Criando modelo de numeros...")
nummodel = LinearSVC(C=1.0,random_state=42)
nummodel.fit(numdata, numlabel)



print("[INFO] Finalizando modelo de numeros...")
f = open(dictio["num"],"wb")
f.write(pickle.dumps(nummodel))
f.close()


import cv2
import numpy as np
import os
from Models import Senia

class Clasificacion():
    sizex = 100
    sizey = 100
    imageSize = sizex * sizey   
    txt_path = "C:\Peals\imagenes"
    svm = None
    clases = []
    __senia = Senia.Senia()
    
    def cargarEntrenamiento(self, nombreArchivo):

        for senia in self.__senia.query().all():
            self.clases.append([int(senia.id), senia.clase])

        self.svm = cv2.ml.SVM_load(nombreArchivo)

    def clasificar(self, img, soloDosClases, claseSoloDosClases):
        respuesta = ""
        # imgAClasificar = self.preProcess(img)
        res = 0

        if img is None:
            return ""
        else:
            res = self.analyseImage(img)
            
            if soloDosClases:
                if res == 1:
                    respuesta = claseSoloDosClases
                else :
                    respuesta = "Otra"
        
        return respuesta
        
        
    def preProcess(self, img):
        return cv2.cvtColor(cv2.resize(img, (self.sizex, self.sizey), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
    
    def analyseImage(self, image):
        image = cv2.resize(image, (self.sizex, self.sizey), interpolation= cv2.INTER_CUBIC)
        sample = np.zeros(shape=(1,self.imageSize), dtype=np.float32)
        
        pos = 0
        width, height = image.shape[:2]
        for w in range(0, width):
            for h in range(0, height):
                pixel = image[w, h]
                sample[0, pos] = pixel
                pos = pos + 1
        return self.svm.predict(sample)[0]

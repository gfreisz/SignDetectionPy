import cv2
import numpy as np
from ImageProcessing import SkinDetector

class Segmentacion():
    filter_maxArea = 70000
    filter_minArea = 6000
    
    mog_refrescar = 0
    mog_refrescarLimite = 20
    
    def __init__(self):
        self.mog_cuerpo = cv2.createBackgroundSubtractorMOG2(500, 16, True)
        self.skin_detector = SkinDetector.SkinDetector()
    
    def updateMog(self, frame):
        fgmask = self.mog_cuerpo.apply(frame, learningRate=0.001)
        ret,imgMog = cv2.threshold(fgmask, 250, 255, cv2.THRESH_TOZERO)
        
        return imgMog;

    def detectarManoSobreCuerpo(self, bgr_frame, binary_frame):
        img_piel = cv2.add(self.skin_detector.HSVDetector(bgr_frame), self.skin_detector.YCCDetector(bgr_frame)) 
        ret_frame = cv2.multiply(img_piel,binary_frame)
        img_coloreada = self.pintarMogCuerpo(ret_frame)
        
        return self.obtenerBorde(bgr_frame, img_coloreada)
        
    def pintarMogCuerpo(self, binary_frame):
        height, width = binary_frame.shape
        
        medioCara = width / 2;
        gray_img = np.zeros((height,width, 1), np.uint8)
        
        cv2.rectangle(gray_img, (0,0), (medioCara, height), 50, -1)
        cv2.rectangle(gray_img, (medioCara,0), (width, height), 150, -1)
        
        invert_frame = (255 - binary_frame)
        invert_frame = cv2.add(invert_frame, np.ones((height,width, 1), np.uint8))
        invert_frame = cv2.multiply(invert_frame, gray_img)
        invert_frame = (255 - invert_frame)
        
        return invert_frame    
    
    def obtenerBorde(self, bgr_frame, binary_frame):
        max_area = 0
        big_contour = None
        height, width = binary_frame.shape
        
        bgr_mano = np.zeros((height,width, 3), np.uint8)
        gray_mano = np.zeros((height,width, 1), np.uint8)
        
        im2, contours, hierarchy = cv2.findContours(binary_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                big_contour = cnt                
                
        if big_contour is not None and 0 < len(big_contour) > max_area > self.filter_minArea:
            cv2.drawContours(bgr_mano, [big_contour], -1, (1,1,1), -1)
            cv2.drawContours(gray_mano, [big_contour], -1, (255), -1)
            
            bgr_mano = cv2.multiply(bgr_mano, bgr_frame)
            gray_mano = cv2.multiply(gray_mano, binary_frame)
            
            edges = cv2.Canny(cv2.cvtColor(bgr_mano, cv2.COLOR_BGR2GRAY), 20, 60)
            gray_mano = cv2.add(gray_mano, edges)
        
            gray_mano = self.separarManoDeBrazo(gray_mano, big_contour)
            
            self.mog_refrescar = 0
        else:
            self.mog_refrescar += 1
            if self.mog_refrescar > self.mog_refrescarLimite:
                self.mog_cuerpo = cv2.createBackgroundSubtractorMOG2(500, 16, True)
                self.mog_refrescar = 0
             
        return gray_mano
    
    def separarManoDeBrazo(self, binary_frame, contour):
        height, width = binary_frame.shape
        binary_mano = binary_frame
        moments = cv2.moments(contour)
        
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        
        hull = cv2.convexHull(contour, returnPoints = False)
        defectos = cv2.convexityDefects(contour, hull)
        
        puntoMasAlejado = None
        ptsSobreCentro = []

        if defectos is not None:
            for i in range(defectos.shape[0]):
                s,e,f,d = defectos[i,0]
                start = tuple(contour[s][0])

                if puntoMasAlejado is None or puntoMasAlejado[1] > start[1]:
                    puntoMasAlejado = start

                if start[1] < cy or abs(cy - start[1]) < 30:
                    ptsSobreCentro.append(start)

            if len(ptsSobreCentro) > 0:
                (x,y),radius = cv2.minEnclosingCircle(np.array(ptsSobreCentro))
                tlx, tly, brx, bry = x - radius, y - radius, x + radius, y + radius

                if tlx < 0: tlx = 0
                if tly < 0: tly = 0
                if brx > width: brx = width
                if bry > height: bry = height

                binary_mano = binary_frame[tly:bry, tlx:brx]
        
        return binary_mano

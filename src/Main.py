import numpy as np
import cv2
from ImageProcessing import Segmentacion
from Cmt import CMT

def main():
    # para evitar error en computadora martin
    cv2.ocl.setUseOpenCL(False)
    
    cap = cv2.VideoCapture(0)
    seg = Segmentacion.Segmentacion()
    
    tlx, tly, brx, bry = -1, -1, -1, -1
    oCmt = CMT.CMT()
    oCmt.estimate_rotation = False
    oCmt.estimate_scale = False
    cmtIniciado = 0
    
    
    
    while(1):
        ret, frame = cap.read()
        
        img = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if (tlx, tly, brx, bry) == (-1, -1, -1, -1):
            # Obtenemos la cara con haarcascade
            face_cascade = cv2.CascadeClassifier('../resources/cascades/haarcascade.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (ax, ay, aw, ah) in faces:
                tlx, tly, brx, bry = ax, ay, ax + aw, ay + ah
                cv2.rectangle(img, (tlx, tly), (brx, bry), (255, 0, 0), 2)
        else:
            # Hacemos seguimiento de la cara
            if cmtIniciado == 1:
                oCmt.process_frame(gray)
                if oCmt.has_result:
                    tlx, tly, brx, bry = oCmt.tl[0], oCmt.tl[1], oCmt.br[0], oCmt.br[1]
                    cv2.rectangle(img, (tlx, tly), (brx, bry), (255, 0, 0), 4)
                    #mask = seg.updateMog(img)
                    #border = seg.detectarManoSobreCuerpo(img, mask)

                    #cv2.imshow('mask', mask)
                    #cv2.imshow('border', border)
                    #cv2.imshow('cmt', img)
            else:
                oCmt.initialise(gray, (tlx, tly), (brx, bry))
                cmtIniciado = 1
                mog_cuerpo = cv2.createBackgroundSubtractorMOG2(500, 16, True)

        cv2.imshow('frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
from Cmt import CMT
from Helpers import Rectangle
from ImageProcessing import Segmentacion
from ImageProcessing import Clasificacion
from PyQt4 import QtCore


class MainController(QtCore.QThread):
    def __init__(self, form):
        super(MainController, self).__init__()
        cv2.ocl.setUseOpenCL(False)

        self.__form = form
        self.__capture = cv2.VideoCapture(0)
        self.__segmentacion = Segmentacion.Segmentacion()
        self.__cmt = None
        self.__rectCara = Rectangle.Rectangle()
        self.__clasificacion = Clasificacion.Clasificacion()

    def run(self):
        self.__clasificacion.cargarEntrenamiento('../resources/SVM/SVM.xml')
        while 1:
            ret, frame = self.__capture.read()
            frame = cv2.flip(frame, 1)
            bgr_img = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            bgr_img = cv2.GaussianBlur(bgr_img, (11, 11), 0)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            if self.__rectCara.isEmpty:
                # Obtenemos la cara con haarcascade
                face_cascade = cv2.CascadeClassifier('../resources/cascades/haarcascade.xml')

                faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

                for (x, y, width, height) in faces:
                    self.__rectCara = Rectangle.Rectangle(x, y, width, height)
            else:
                # Hacemos seguimiento de la cara
                if self.__cmt is None:
                    self.__cmt = CMT.CMT()
                    self.__cmt.estimate_rotation = False
                    self.__cmt.estimate_scale = False
                    if self.__cmt.initialise(gray_img, self.__rectCara.topLeft, self.__rectCara.bottomRight) is False:
                        self.__cmt = None
                else:
                    self.__cmt.process_frame(gray_img)

                    if self.__cmt.has_result:
                        self.__rectCara = Rectangle.Rectangle(self.__cmt.tl[0], self.__cmt.tl[1], self.__cmt.tr[0] - self.__cmt.tl[0], self.__cmt.bl[1] - self.__cmt.tl[1])
                        mask = self.__segmentacion.updateMog(bgr_img)
                        border = self.__segmentacion.detectarManoSobreCuerpo(bgr_img, mask)

                        self.emit(QtCore.SIGNAL("SALIDAS_IMG"), mask, border)

                        if self.__form.ui.cbtn_clasificar.isChecked():
                            self.__clasificacion.clasificar(border, False, "")

                cv2.rectangle(bgr_img, self.__rectCara.topLeft, self.__rectCara.bottomRight, (255, 0, 0), 2)

            self.emit(QtCore.SIGNAL("BGR_IMG"), cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

    def dispose(self):
        self.__capture.release()
        self.quit()

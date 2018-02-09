from PyQt4 import QtGui, QtCore
from form_main_designer import Ui_MainWindow
from Controllers import MainController
from Helpers import Utils

class form_main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.__mainController = MainController.MainController(self)
        self.connect(self.__mainController, QtCore.SIGNAL("BGR_IMG"), self.GUI_updateEntrada)
        self.connect(self.__mainController, QtCore.SIGNAL("SALIDAS_IMG"), self.GUI_updateSalida)
        self.__mainController.start()

    def GUI_updateEntrada(self, cvImage):
        pixmap_img = Utils.convertMatToQPixmap(cvImage, 3)
        pixmap_img = pixmap_img.scaled(self.ui.img_entrada.size())
        self.ui.img_entrada.setPixmap(pixmap_img)

    def GUI_updateSalida(self, cvMask, cvBorder):
        pixmap_mask = Utils.convertMatToQPixmap(cvMask, 1)
        pixmap_mask = pixmap_mask.scaled(self.ui.img_resultado.size())

        pixmap_border = Utils.convertMatToQPixmap(cvBorder, 1)
        pixmap_border = pixmap_border.scaled(self.ui.img_mogColoreado.size())

        self.ui.img_resultado.setPixmap(pixmap_mask)
        self.ui.img_mogColoreado.setPixmap(pixmap_border)

    def closeEvent(self, event):
        self.__mainController.dispose()

        self.deleteLater()
        event.accept()

from PyQt4.QtGui import QPixmap, QImage

def convertMatToQImage(cvImage, channels):
    height, width = cvImage.shape[:2]
    bytesPerLine = channels * width
    qFormat = QImage.Format_ARGB32 if channels == 4 else (QImage.Format_RGB888 if channels == 3 else QImage.Format_Indexed8)

    qImg = QImage(cvImage.data, width, height, bytesPerLine, qFormat)

    return qImg

def convertMatToQPixmap(cvImage, channels):
    qImg = convertMatToQImage(cvImage, channels)
    qPixmap = QPixmap(qImg)

    return qPixmap

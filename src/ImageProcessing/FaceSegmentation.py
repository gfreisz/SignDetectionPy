import cv2
import numpy as np
from ImageProcessing import SkinDetector
from Helpers import Rectangle

class FaceSegmentation:
    def __init__(self, width, height, smooth, erode, dilate, aperture, close):
        self.__width = width
        self.__height = height
        self.__smooth = smooth
        self.__erode = erode
        self.__dilate = dilate
        self.__aperture = aperture
        self.__close = close

        self.__mog_cara = cv2.createBackgroundSubtractorMOG2(500, 16, True)
        self.__skin_detector = SkinDetector.SkinDetector()
        self.__img_prev = None
        self.__nose_center = [-1, -1]
        self.__cuads = [-1, -1, -1, -1]

    def procesar_cara(self, bgr_cara, pt_nariz):
        self.__nose_center = pt_nariz
        w, h = bgr_cara.shape()[:2]
        self.calculate_cuads(w, h)
        mask = self.preprocesar_imagen(bgr_cara)

        if mask is not None:
            im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            big_contour = None
            max_area = -1
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    big_contour = cnt

            if big_contour is not None and 0 < len(big_contour) > max_area > 300:
                img_background = np.zeros((mask.shape()[0], mask.shape()[1], 4), np.uint8)
                img_tmp = cv2.cvtColor(bgr_cara, cv2.COLOR_BGR2BGRA)
                img_foreground = np.zeros((bgr_cara.shape()[0], bgr_cara.shape()[1], 4), np.uint8)

                cv2.drawContours(img_background, [big_contour], -1, (1,1,1,1), -1)

                img_tmp_2 = cv2.multiply(img_background, img_tmp)
                img_background = cv2.GaussianBlur(img_tmp_2, (11, 11))
                img_background = cv2.Canny(img_background, 20, 20)
                img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2BGRA)
                img_background = cv2.add(img_tmp, img_background)

                #if (contour.BoundingRectangle.IntersectsWith(cuadrantes[0]))
                #    img_foreground.Draw(cuadrantes[0], new Rgba(255, 0, 0, 0.2), -1);
                #if (contour.BoundingRectangle.IntersectsWith(cuadrantes[1]))
                #    img_foreground.Draw(cuadrantes[1], new Rgba(0, 255, 0, 0.2), -1);
                #if (contour.BoundingRectangle.IntersectsWith(cuadrantes[2]))
                #    img_foreground.Draw(cuadrantes[2], new Rgba(0, 0, 255, 0.2), -1);
                #if (contour.BoundingRectangle.IntersectsWith(cuadrantes[3]))
                #    img_foreground.Draw(cuadrantes[3], new Rgba(255, 255, 0, 0.2), -1);

                #return img_background.Mul(img_foreground).Convert<Gray, byte>();
                return img_background

        return np.zeros(bgr_cara.shape(), np.uint8)

    def calculate_cuads(self, width, height):
        c_x = self.__nose_center[0]
        c_y = self.__nose_center[1]

        self.__cuads[0] = Rectangle.Rectangle(0, 0, c_x, c_y);
        self.__cuads[1] = Rectangle.Rectangle(c_x, 0, width - c_x, c_y);
        self.__cuads[2] = Rectangle.Rectangle(c_x, c_y, width - c_x, height - c_y);
        self.__cuads[3] = Rectangle.Rectangle(0, c_y, c_x, height - c_y);

    def preprocesar_imagen(self, bgr_roi):
        bgr_skin = cv2.add(self.skin_detector.HSVDetector(bgr_roi), self.skin_detector.YCCDetector(bgr_roi))
        bgr_smooth = cv2.GaussianBlur(bgr_roi, (self.__smooth, self.__smooth), 0)

        fg_mask = self.mog_cuerpo.apply(bgr_smooth, learningRate=0.001)
        fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_TOZERO)
        fg_mask = cv2.resize(fg_mask, bgr_skin.shape[:2], cv2.INTER_LINEAR)
        fg_mask = cv2.multiply(fg_mask, bgr_skin)

        if self.__img_prev is not None:
            bgr_smooth = cv2.resize(bgr_smooth, self.__img_prev.shape[:2], cv2.INTER_LINEAR)
            gray_img = cv2.cvtColor(bgr_smooth, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.absdiff(self.__img_prev, gray_img)

            tmp = np.zeros(gray_img.shape, np.int8)
            tmp[:] = 40

            gray_img = cv2.multiply(gray_img, tmp)
            gray_img = cv2.threshold(gray_img, 240, 255)

            nonZeros = cv2.countNonZero(gray_img)[:1]
            if nonZeros > (self.__width * self.__height * .25):
                fg_mask = cv2.multiply(fg_mask, gray_img)

        return fg_mask

    def applyOpMorph(self, img):
        if (self.__aperture):
            img = cv2.erode(img, self.__erode)
            img = cv2.dilate(img, self.__dilate)
        if (self.__close):
            img = cv2.dilate(img, self.__dilate)
            img = cv2.erode(img, self.__erode)

        return img

    def reset(self):
        self.__mog_cara = cv2.createBackgroundSubtractorMOG2(500, 16, True)

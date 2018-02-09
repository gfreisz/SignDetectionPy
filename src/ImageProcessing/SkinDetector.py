import numpy as np
import cv2

class SkinDetector():
    hsv_min = np.array([0,45,0])
    hsv_max = np.array([20,255,255])
    
    ycc_min = np.array([0,131,80])
    ycc_max = np.array([255,185,135])
    
    def HSVDetector(self, bgr_frame):
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_frame, self.hsv_min, self.hsv_max)
    
    def YCCDetector(self, bgr_frame):
        ycc_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        return cv2.inRange(ycc_frame, self.ycc_min, self.ycc_max)
# 2019 Adam Szkolny <adamszkolny@gmail.com>.
import cv2 as cv
import numpy as np


class SudokuWarper:
    def __init__(self):
        pass

    def __init__(self, maxValue):
        self.maxValue = maxValue

    def __findMaxContour(self, contours):
        if len(contours) == 0:
            return None
        else:
            maxContourArea = -1
            maxContour = contours[0]
            for c in contours:
                if cv.contourArea(c) > maxContourArea:
                    maxContourArea = cv.contourArea(c)
                    maxContour = c
            return maxContour

    def __getWarpedImage(self, image, contour):
        peri = cv.arcLength(contour, True)
        contour = cv.approxPolyDP(contour, 0.02 * peri, True)
        
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
    
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv.getPerspectiveTransform(rect, dst)
        warp = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warp

    def warpImage(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (9, 9), 0)
        gray = cv.medianBlur(gray, 3)
        thresholded = cv.adaptiveThreshold(gray, self.maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, 11, 2)
        kernel = np.ones((5,5))

        for i in range(2):
            thresholded = cv.erode(thresholded, kernel, iterations = 1)
            thresholded = cv.dilate(thresholded, kernel, iterations = 1)

        edges = cv.Canny(thresholded,0, 255)
        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        c = self.__findMaxContour(contours)
        image = cv.drawContours(image, [c], -1, (0, 255, 0), 1)
        image = self.__getWarpedImage(thresholded, c)

        return image

        


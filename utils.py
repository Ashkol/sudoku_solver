# 2019 Adam Szkolny <adamszkolny@gmail.com>.

import cv2 as cv
import numpy as np

def histogramEqualize(image):
    image = cv.imread('test_010.jpg')

    image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    channels = cv.split(image)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, image)
    return cv.cvtColor(image, cv.COLOR_YUV2BGR)
    

def fillHoles(image):
    des = cv.bitwise_not(image)
    contours, hier = cv.findContours(des,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        cv.drawContours(des,[c],0,255,-1)

    return cv.bitwise_not(des)


def deleteEdges(image, xWidth, yWidth):
    '''Returns image with deleted edges'''
    height, width = image.shape[0], image.shape[1]
    cropImg = image[yWidth:height-yWidth, xWidth:width-xWidth]
    return cropImg


def __getUpperLeftContourCorner(contour):
    pts = __reshapeContour2Rect(contour)
    s = pts.sum(axis = 1)
    return pts[np.argmin(s)]


def sortContours(image, contours):
    '''Sort rectangular contours by their upper-left corners'''
    weight = image.shape[1] * 0.005
    return sorted(contours, key = lambda contour: (__getUpperLeftContourCorner(contour)[0] + __getUpperLeftContourCorner(contour)[1]*weight), reverse = True)


def __reshapeContour2Rect(contour):
    peri = cv.arcLength(contour, True)
    contour = cv.approxPolyDP(contour, 0.02 * peri, True)
    return contour.reshape(4, 2)


def findMaxContour(contours):
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

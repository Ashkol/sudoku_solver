# 2019 Adam Szkolny <adamszkolny@gmail.com>.
import argparse
import cv2 as cv
import numpy as np
import pytesseract
import sudoku_warper, utils


thr_min, thr_max = 130,  255
kernel = np.ones((3, 3), np.uint8)

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

   
def getWarpedImage(image, contour):
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


def getAvgLuminance(image):
    sigma = np.average(image)
    return  sigma


def getNoGridBoard(image, contours, kernel):
    global thr_min, thr_max
    cells = []
    whitespaces = []
    for c in contours[::-1]:
        warpedCell = getWarpedImage(image, c)
        warpedCell = cv.resize(warpedCell, (200, 200))
        warpedCell = utils.deleteEdges(warpedCell, 20, 20)
        _, warpedCell = cv.threshold(warpedCell, thr_min, thr_max, cv.THRESH_BINARY)
        warpedCell = cv.erode(warpedCell, kernel, iterations = 1)
        warpedCell = cv.dilate(warpedCell, kernel, iterations = 1)
        print(getAvgLuminance(warpedCell))
        cv.imshow("CELL", warpedCell)
        cv.waitKey(2)
        if getAvgLuminance(warpedCell) < (thr_max - 12):
            whitespaces.append(1)
            cells.append(warpedCell)
        else:
            whitespaces.append(0)

    print(len(cells))
    print(whitespaces)
    cv.waitKey(0)
        
    return cells, whitespaces

def mergeWithBooleanMask(numbers, mask, nullSign = -1):
    sudokuList = []
    j = 0
    for i in range(0, len(mask)):
        if mask[i]:
            sudokuList.append(int(numbers[j]))
            j += 1
        else:
            sudokuList.append(nullSign)
    return sudokuList


def filterContours(contours, numberOfContours, deviation):
    retContours = []
    tempContours = sorted(contours, key = cv.contourArea, reverse = True)
    if cv.contourArea(contours[0]) > 2 * cv.contourArea(contours[1]):
        for c in tempContours:
            if cv.contourArea(c) == cv.contourArea(tempContours):
                contours.remove(c)
                break
    for c in contours:
        if cv.contourArea(c) > cv.contourArea(tempContours[1]) * (1-deviation):
            retContours.append(c)
            
    return retContours


def main():
    global thr_min, thr_max, kernel

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='imgFile', required=True, help='path to the input image')
    args = parser.parse_args()

    image = cv.imread(args.imgFile)

    warper = sudoku_warper.SudokuWarper(255)
    sudokuBoard = warper.warpImage(image)
    cv.namedWindow('Sudoku Board', cv.WINDOW_NORMAL)
    cv.imshow("Sudoku Board", sudokuBoard)
    cv.waitKey(0)
    
    contours, _ = cv.findContours(sudokuBoard, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = filterContours(contours, 81, 0.2)
    contours = utils.sortContours(sudokuBoard, contours)

    kernel = np.ones((3,3),np.uint8)

    width, height = 0, 0
    for c in contours:
        height += c.shape[0]
        width += c.shape[1]

    contourBoard = cv.cvtColor(sudokuBoard, cv.COLOR_GRAY2BGR)
    for c in contours:
       contourBoard = cv.drawContours(contourBoard, [c], -1, (0, 0, 255), 1)

    print(len(contours))
    cv.namedWindow('Contours', cv.WINDOW_NORMAL)
    cv.imshow("Contours", contourBoard)
    cv.waitKey(0)
    
    cv.destroyAllWindows()

    cells, whitespaces = getNoGridBoard(sudokuBoard, contours, kernel)
    sudokuNumbers = []
    
    for c in cells:
        #c = cv.erode(c, np.ones((5,5),dtype='uint8'), iterations = 1)
        #c = cv.dilate(c, np.ones((5,5),dtype='uint8'), iterations = 1)
        c = cv.medianBlur(c, 5)
        number = pytesseract.image_to_string(c, config='-psm 10 outputbase digits')
        if not number.isnumeric():
            number = pytesseract.image_to_string(utils.fillHoles(c), config='-psm 10 outputbase digits')
            print("Filled")
            c = utils.fillHoles(c)
        print(number)
        sudokuNumbers.append(number)
        cv.imshow("Cell", c)
        cv.waitKey(0)

    print("Merged")
    print(mergeWithBooleanMask(sudokuNumbers, whitespaces))
    
      
                            
if __name__ == "__main__":
    main()

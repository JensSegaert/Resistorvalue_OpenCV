#########################################################################################################################
""""
Author: Jens Segaert
Date: 03/06/2022
"""
#########################################################################################################################

# Import Libraries
import cv2
import numpy as np
import os


DEBUG = False
COLOUR_BOUNDS = [
    [(0, 0, 0), (179, 255, 93), "BLACK", 0, (0, 0, 0)],
    [(0, 90, 10), (15, 250, 100), "BROWN", 1, (0, 51, 102)],
    [(0, 30, 80), (10, 255, 200), "RED", 2, (0, 0, 255)],
    [(10, 70, 70), (25, 255, 200), "ORANGE", 3, (0, 128, 255)],
    [(30, 170, 100), (40, 250, 255), "YELLOW", 4, (0, 255, 255)],
    [(35, 20, 110), (60, 45, 120), "GREEN", 5, (0, 255, 0)],
    [(65, 0, 85), (115, 30, 147), "BLUE", 6, (255, 0, 0)],
    [(120, 40, 100), (140, 250, 220), "PURPLE", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 90), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
];

# Define parameters for using haarcascade in script (detectMultiScale)
scale_factor = 1.1
min_neighbors = 25
min_size = (30, 30)

# For red mask
RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)

# Define min area for contour validation
MIN_AREA = 700
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Required for trackbars
def empty(x):
    pass


# Initializing haar cascade and video source
def init(DEBUG, path):
    if (DEBUG):
        cv2.namedWindow("frame")
        cv2.createTrackbar("lh", "frame", 0, 179, empty)
        cv2.createTrackbar("uh", "frame", 0, 179, empty)
        cv2.createTrackbar("ls", "frame", 0, 255, empty)
        cv2.createTrackbar("us", "frame", 0, 255, empty)
        cv2.createTrackbar("lv", "frame", 0, 255, empty)
        cv2.createTrackbar("uv", "frame", 0, 255, empty)
    resClose = []
    
    # Read image
    img = cv2.imread(path)

    # Convert to grayscale
    gliveimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load resistor haar cascade classifier
    rectcascade = cv2.CascadeClassifier(os.path.abspath(os.getcwd()) + "\cascade\haarcascade_resistors_0.xml")
    ressfind = rectcascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    if len(ressfind) != 0:
        # create the bounding box around the detected resistor
        for (x, y, w, h) in ressfind:
            roi_gray = gliveimg[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Apply secondPass to filter false positives
            secondPass = rectcascade.detectMultiScale(roi_gray, 1.01, 5)
            if (len(secondPass) != 0):
                resClose.append((np.copy(roi_color), (x, y, w, h)))
    
    # Show image of resistor detection
    cv2.imshow("Resistor Detection", img)
    
    k = cv2.waitKey(30)
    if k == 27:  # Wait for ESC key to exit
        cv2.destroyAllWindows()
    print(resClose)
    
    return (img, rectcascade)


# Returns true if contour is valid, false otherwise
def validContour(cnt):
    # Looking for a large enough area and correct aspect ratio
    if (cv2.contourArea(cnt) < MIN_AREA):
        return False
    else:
        x, y, w, h = cv2.boundingRect(cnt)
        aspectRatio = float(w) / h
        if (aspectRatio > 0.4):
            return False
    return True




# Uses haar cascade to identify resistors in the image
def findResistors(img, rectCascade):
    gliveimg = cv2.cvtColor(cliveimg, cv2.COLOR_BGR2GRAY)
    resClose = []

    # Detect resistors in main frame
    ressFind = rectCascade.detectMultiScale(gliveimg, 1.1, 25)
    for (x, y, w, h) in ressFind:  # SWITCH TO H,W FOR <CV3

        roi_gray = gliveimg[y:y + h, x:x + w]
        roi_color = cliveimg[y:y + h, x:x + w]

        # Apply another detection to filter false positives
        secondPass = rectCascade.detectMultiScale(roi_gray, 1.01, 5)

        if (len(secondPass) != 0):
            resClose.append((np.copy(roi_color), (x, y, w, h)))
    
    return resClose

list_bandspos = []

# Analysis close up image of resistor to identify bands
def findBands(resistorInfo, DEBUG):
    if (DEBUG):
        uh = cv2.getTrackbarPos("uh", "frame")
        us = cv2.getTrackbarPos("us", "frame")
        uv = cv2.getTrackbarPos("uv", "frame")
        lh = cv2.getTrackbarPos("lh", "frame")
        ls = cv2.getTrackbarPos("ls", "frame")
        lv = cv2.getTrackbarPos("lv", "frame")

    # Enlarge image
    resImg = cv2.resize(resistorInfo[0], (400, 200))
    print('resistorinfo')
    print(resistorInfo[0])

    # Show image of close-up
    cv2.imshow('resistor_close_up', resImg)

    # Save resistor close up
    resPos = resistorInfo[1]

    # Apply bilateral filter and convert to hsv
    pre_bil = cv2.bilateralFilter(resImg, 15, 80, 80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)

    # Edge threshold filters out background and resistor body
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 59, 5)
    thresh = cv2.bitwise_not(thresh)
    
    # Show tresholded image of close-up
    cv2.imshow('resistor_close_up_tresholded', thresh)


    bandsPos_left = []
    bandsPos_right = []

    # If in debug mode, check only one colour
    if (DEBUG):
        checkColours = COLOUR_BOUNDS[0:1]
    else:
        checkColours = COLOUR_BOUNDS

    for clr in checkColours:
        if (DEBUG):
            mask = cv2.inRange(hsv, (lh, ls, lv), (uh, us, uv))  # use trackbar values
        else:
            mask = cv2.inRange(hsv, clr[0], clr[1])
            if (clr[2] == "RED"):  # combining the 2 RED ranges in hsv
                redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
                mask = cv2.bitwise_or(redMask2, mask, mask)
        
        
        mask = cv2.bitwise_and(mask, thresh, mask=mask)
        
        # Find contours
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # Filter invalid contours, store valid ones
        for k in range(len(contours) - 1, -1, -1):
            print(len(contours))
            if (validContour(contours[k])):
                print('contours')
                print(contours)
                leftmostPoint = tuple(contours[k][contours[k][:, :, 0].argmin()][0])
                rightmostPoint = tuple(contours[k][contours[k][:, :, 0].argmax()][0])
                bandsPos_left += [leftmostPoint]
                bandsPos_right += [rightmostPoint]
                cv2.circle(pre_bil, leftmostPoint, 5, (255, 0, 255), -1)
            else:
                contours.pop(k)

        # Draw contours
        cv2.drawContours(pre_bil, contours, -1, clr[-1], 3)

        # Show mask and tresholded image 
        if (DEBUG):
            cv2.imshow("mask", mask)
            cv2.imshow('thresh', thresh)

    cv2.imshow('Contour Display', pre_bil)  # Shows the most recent resistor checked.
    
    print('sorted bandspos_left')
    print(sorted(bandsPos_left))
    print('sorted bandspos_right')
    print(sorted(bandsPos_right))
    print('contours')
    print(contours)

    return sorted(bandsPos_left), sorted(bandsPos_right)







# Define path of input image
path = os.path.abspath(os.getcwd()) + "\input_images\img_4.png"


# Call functions
img, rectCascade = init(DEBUG, path)
while (not (cv2.waitKey(1) == ord('q'))):
    cliveimg = cv2.imread(path)
    resClose = findResistors(cliveimg, rectCascade)
    for i in range(len(resClose)):
        bandsPos_left, bandsPos_right = findBands(resClose[i], DEBUG)

    cv2.imshow("Frame", cliveimg)


cv2.destroyAllWindows()

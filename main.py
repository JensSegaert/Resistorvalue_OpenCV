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
from scipy.spatial import distance
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance

# Predefine DEBUG as False
DEBUG = False

# List colour bounds
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

# For red mask
RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Edit: change 'MIN_AREA' to set minimum area for contour validation of resistorbands
MIN_AREA = 700

# Path to save all images from running code in folder 'images_code'
path_images_code = os.path.abspath(os.getcwd()) + '\images_code'

# Path of image input of resistor
path_image = os.path.abspath(os.getcwd()) + "\input_images\img_4.png"

# Path to save result
path_result = os.path.abspath(os.getcwd()) + '\\' + 'result'

# Path for training
path_for_training = os.path.abspath(os.getcwd()) + "\images_training_specific_resistorfactory_no-artificial-lighting"



# Required function for trackbars in function 'init'
def empty(x):
    pass


# Initializing haar cascade and image input
def init(DEBUG, path_image):
    print('begin init')
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
    img = cv2.imread(path_image)

    # Convert to grayscale
    gliveimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load resistor haar cascade classifier
    rectcascade = cv2.CascadeClassifier(os.path.abspath(os.getcwd()) + "\cascade\haarcascade_resistors_0.xml")

    # Define parameters detectMultiScale
    scale_factor = 1.1
    min_neighbors = 25
    min_size = (30, 30)

    ressfind = rectcascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    if len(ressfind) != 0:
        # Create the bounding box around the detected resistor
        for (x, y, w, h) in ressfind:
            roi_gray = gliveimg[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            
            # Create rectangle on image 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Apply another detection to filter false positives
            secondPass = rectcascade.detectMultiScale(roi_gray, 1.01, 25)
            if (len(secondPass) != 0):
                resClose.append((np.copy(roi_color), (x, y, w, h)))

    # Show image of detected resistor 
    cv2.imshow("Resistor Detection", img)
    
    # Save image of detected resistor in folder 'images_code'
    cv2.imwrite(os. path. join(path_images_code , 'resistor_detection_by_haarcascade.jpg'), img)

    return (img, rectcascade)


# Returns true if contour is valid, false otherwise
def validContour(cnt):
    print('begin validContour')

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
    print('begin findResistors')
    
    gliveimg = cv2.cvtColor(cliveimg, cv2.COLOR_BGR2GRAY)
    resClose = []

    # Detect resistors in main frame
    ressFind = rectCascade.detectMultiScale(gliveimg, 1.1, 25)
    for (x, y, w, h) in ressFind:  # SWITCH TO H,W FOR <CV3

        roi_gray = gliveimg[y:y + h, x:x + w]
        roi_color = cliveimg[y:y + h, x:x + w]

        # Apply another detection to filter false positives
        secondPass = rectCascade.detectMultiScale(roi_gray, 1.01, 25)

        if (len(secondPass) != 0):
            resClose.append((np.copy(roi_color), (x, y, w, h)))

    return resClose, x, y, w, h



# Analysis close up image of resistor to identify bands
def findBands(resistorInfo, DEBUG):
    print('begin findBands')
    if (DEBUG):
        uh = cv2.getTrackbarPos("uh", "frame")
        us = cv2.getTrackbarPos("us", "frame")
        uv = cv2.getTrackbarPos("uv", "frame")
        lh = cv2.getTrackbarPos("lh", "frame")
        ls = cv2.getTrackbarPos("ls", "frame")
        lv = cv2.getTrackbarPos("lv", "frame")

    # Enlarge image
    resImg = cv2.resize(resistorInfo[0], (400, 200))

    # Save image of close-up
    cv2.imwrite(os.path.join(path_images_code, 'resistor_close-up.jpg'), resImg)

    resPos = resistorInfo[1]


    # Apply bilateral filter 
    pre_bil = cv2.bilateralFilter(resImg, 15, 80, 80)
    
    # Convert to hsv
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    
    # Save image of bilateral filter in folder 'images_code'
    cv2.imwrite(os.path.join(path_images_code, 'bilateral-filter.jpg'), pre_bil)

    # Edge threshold filters out background and resistor body
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 59, 5)
    thresh = cv2.bitwise_not(thresh)

    # Show tresholded image of resistor close-up 
    cv2.imshow('resistor_close-up_tresholded', thresh)
    
    # Save image in folder 'images_code'
    cv2.imwrite(os. path. join(path_images_code , 'resistor_close-up_tresholded.jpg'), thresh)

    bandsPos_left = []
    bandsPos_right = []

    # If in debug mode, check only one colour
    if (DEBUG):
        checkColours = COLOUR_BOUNDS[0:1]
    else:
        checkColours = COLOUR_BOUNDS

    for clr in checkColours:
        if (DEBUG):
            mask = cv2.inRange(hsv, (lh, ls, lv), (uh, us, uv))  # Use trackbar values
        else:
            mask = cv2.inRange(hsv, clr[0], clr[1])
            if (clr[2] == "RED"):  # Combining the 2 RED ranges in hsv
                redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
                mask = cv2.bitwise_or(redMask2, mask, mask)
        
        
        mask = cv2.bitwise_and(mask, thresh, mask=mask)
        
        # Find contours of colorbands
        im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print('contours')
        print(contours)
        
        # Filter invalid contours, store valid ones
        for k in range(len(contours) - 1, -1, -1):
            print(len(contours))
            
            if (validContour(contours[k])):
                
                # Get Leftmostpoint of each contour
                Leftmostpoint = tuple(contours[k][contours[k][:, :, 0].argmin()][0])
                
                # Get Rightmostpoint of each contour
                Rightmostpoint = tuple(contours[k][contours[k][:, :, 0].argmax()][0])
                
                # Add Leftmostpoint to list
                bandsPos_left += [Leftmostpoint]
                
                # Add Rightmostpoint to list
                bandsPos_right += [Rightmostpoint]
               
                cv2.circle(pre_bil, Leftmostpoint, 5, (255, 0, 255), -1)
            else:
                contours.pop(k)

        # Draw contours
        cv2.drawContours(pre_bil, contours, -1, clr[-1], 3)

        # Show mask and tresholded image
        if (DEBUG):
            cv2.imshow("mask", mask)
            cv2.imshow('thresh', thresh)

    # Show contour display image
    cv2.imshow('Contour Display', pre_bil)  # Shows the most recent resistor checked.
    
    # Save image of colorband-contours in folder 'images_code'
    cv2.imwrite(os. path. join(path_images_code , 'contour_color_bands.jpg'), pre_bil)

    print('sorted bandspos_left')
    print(sorted(bandsPos_left))
    print('sorted bandspos_right')
    print(sorted(bandsPos_right))
    print('contours')
    print(contours)

    return sorted(bandsPos_left), sorted(bandsPos_right)


# Define function for clustertraining of color images
def training_clustering(path_for_training):
        print('begin training kmeans')
        
        """
           Description: take all pictures in directory 'Images_training_specific_resistorfactory_no-artificial-lighting'
           --> take mean BGR pixel value of each picture
           --> cluster all mean RGB-values of each picture
        """
        
        dirs = os.listdir(path_for_training)
        BGR_list = []
        BR_list = []
        
        # Loop in directory for images
        for item in dirs:
            
            # Define fullpath of items in training
            fullpath = os.path.join(path_for_training, item)
            
            if os.path.isfile(fullpath):
                img = np.array(Image.open(fullpath))
                print('foto')

                # print name_picture:: name of picture = string of fullpath substracted from path string to get name of image itself
                name_picture = fullpath.replace(path_for_training, '')

                # remove '\' from image name just for printing
                name_picture = name_picture.replace('\\', '')
                print(name_picture)

                # Read image
                BGR_training_img = cv2.imread(fullpath)

                # Calculate mean BGR pixel value of image 
                avg_color_per_row = np.average(BGR_training_img, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)

                print('mean_BGR_value_picture')
                print(avg_color)
                BGR_list.append(list(avg_color))

                # Make also BR list used for 2D plot of blue and red (green not involved).
                avg_color_BR = avg_color.copy()
                avg_color_BR = list(avg_color_BR)

                # Delete green value from BGR-list and we have BR-list for mean value of each image
                del avg_color_BR[1]

                # Add each mean BR-value to a list
                BR_list.append(avg_color_BR)

                print('BGR_list for now')
                print(BGR_list)
                print('BR_list for now')
                print(BR_list)

        # Cluster mean BGR-values images
        # n_clusters is the number of clusters you want to use to classify your data
        kmeans_BGR = KMeans(n_clusters=10, random_state=0).fit(BGR_list)

        # See the labels with:
        print('labels_kmeans')
        print(kmeans_BGR.labels_)

        # See were the centres of your clusters are (BGR)
        print(kmeans_BGR.cluster_centers_)

        list_cluster_centers_BGR = kmeans_BGR.cluster_centers_
        print('list_cluster_centers')
        print(list_cluster_centers_BGR)


        # Return list of cluster centers of BGR-values
        return list_cluster_centers_BGR


# Get center rectangle of each colorband
def get_color_bands(Left, Right, BGR_list):
    print('begin get_color_bands')

    # Check if we have 3 resistorband-contours as we need. If less --> stop function,
    # If more --> delete leftmostpoints and rightmostpoints of contours defined after 3 first contours
    if len(Left) > 3:
        del Left[3:]
    if len(Left) < 3:
        print('must have 3 contours to detect resistor')
        return    # Stop function
    else: # so if there are 3 contours, change nothing
        Left = Left

    if len(Right) > 3:
        del Right[3:]
    if len(Right) < 3:
        print('must have 3 contours to detect resistor')
        return    # Stop function
    else: # so if there are 3 contours, change nothing
        Right = Right
    
    print('Left')
    print(Left)
    print('Right')
    print(Right)


    Contour_center_list = []

    # Read image of resistor_close_up
    resistor_close_up_for_rectangles_bands = cv2.imread(os.path.abspath(os.getcwd()) + "\images_code\bilateral-filter.jpg")


    # Line 380-390: Adjust close up image of resistor with color balance
    
    # Open the image
    img = Image.open(os.path.abspath(os.getcwd()) + "\images_code' + '\\' +  'bilateral-filter.jpg")

    # Adjust image color balance
    enhancer = ImageEnhance.Color(img)

    # Image color balance factor:
    factor = 2.5

    # Output editted input image
    editted_output = enhancer.enhance(factor)
    
    # Save editted image in folder 'images_code'
    image = editted_output.save(f"{path_images_code}\\editted_image.png")
    
    # Read editted image
    resistor_close_up_for_rectangles_bands = cv2.imread(os.path.abspath(os.getcwd()) +  '\images_code\editted_image.png')
    
    # Make a copy of image
    copy_resistor_clean = resistor_close_up_for_rectangles_bands.copy()

    # Get center point of each contour by taking mean value of x and y : [(x1+x2)/2 , (y1+y2)/2]
    # Add the (x,y)-coördinate of each contour center to a list
    for contour_number in range(len(Left)):
        Leftmostpoint_single_contour_x = Left[contour_number][0]
        Rightmostpoint_single_contour_x = Right[contour_number][0]
        Leftmostpoint_single_contour_y = Left[contour_number][1]
        Rightmostpoint_single_contour_y = Right[contour_number][1]
        print(Leftmostpoint_single_contour_x)
        Contour_center_list.append([[(Leftmostpoint_single_contour_x + Rightmostpoint_single_contour_x)/2] , [(Leftmostpoint_single_contour_y + Rightmostpoint_single_contour_y)/2]])
    
    print('Contour_center_list')
    print(Contour_center_list)

    list_avg_color_contours = []

    # Pop the x and y-value of each contour center
    for contour_center in range(len(Contour_center_list)):
        BandClose = []
        
        # Get x-value contour center
        x = [round(float(s)) for s in Contour_center_list[contour_center][0]] # still a list, must be integer
        x = x.pop()
        
        # Get y-value contour center
        y = [round(float(s)) for s in Contour_center_list[contour_center][1]] # still a list, must be integer
        y = y.pop()
        

        # Determine left, right, top and bottom for rectangle of center of resistorband
        left = x - 3
        bottom = y
        right = x + 3
        top = y - 15
        
        # Create rectangles of contourcenters
        rectangle_colorband = cv2.rectangle(resistor_close_up_for_rectangles_bands, (left,bottom),(right,top),(255,255,255),1)

        # Get new image of contour center by cropping the center rectangle
        im2 = copy_resistor_clean[top: bottom, left: right]
        cv2.imwrite(os.path.join(path_images_code, 'resistor_center_band_close_up' + str(contour_center) + '.jpg'), im2)

        # Show the image of the contourcenter and save it in folder 'images_code'
        cv2.imshow('Band_contour_center', rectangle_colorband)
        cv2.imwrite(os.path.join(path_images_code, 'band_contour_centers.jpg'), rectangle_colorband)

        # Calculate mean BGR value of contour center rectangle image
        avg_color_per_row_contourcenter = np.average(im2, axis=0)
        avg_color_center_contour = np.average(avg_color_per_row_contourcenter, axis=0)
        tuple_avg_color_center_contour = tuple(avg_color_center_contour)
        list_avg_color_contours.append(list(avg_color_center_contour))
        
        print('avg_color_center_contour')
        print(tuple_avg_color_center_contour)
        print(list_avg_color_contours)


    # Make list with colors
    list_colors = ['Green', 'White', 'Orange', 'Gray', 'Black', 'Yellow', 'Red',  'Purple', 'Blue', 'Brown']

    # This is going to be the list of the colors of the bands
    # Par example, if we get ['Green','Black','Red'], it means color of band 1: green, band 2: black and band 3: red
    color_list_bands = []

    # Calculate norm between the centerband value and each color cluster center point (from training python files) in BGR
    # Do this for all contour bands
    for u in range(0,len(list_avg_color_contours)):
      list_distances = []
        for w in range(0,len(BGR_list)):
          dst = distance.euclidean(tuple(BGR_list[w]), tuple(list_avg_color_contours[u]))
          
          list_distances.append(dst)
          

      # Go off distance list and look for minimum distance,
      # Link this to the color and put color string of list_colors in 'color_list_bands'
      for l in range(0,len(list_distances)):
             print('cycle')
             if list_distances[l] == min(list_distances):
                color_list_bands.append(list_colors[l])


    print('color_list_bands')
    print(color_list_bands)

    return color_list_bands



# Function to calculate resistorvalue with information of the color the bands possess.
def calculate_result(color_list_bands):
    print('begin calculating result')

    # Create an empty string for the first 2 colorbands
    string_first_2bands = ''

    # Make integer string of values of first 2 bands
    for y in range(0,len(color_list_bands)-1):
        if color_list_bands[y] == 'Black':
            string_first_2bands += '0'
        if color_list_bands[y] == 'Brown':
            string_first_2bands += '1'
        if color_list_bands[y] == 'Red':
            string_first_2bands += '2'
        if color_list_bands[y] == 'Orange':
            string_first_2bands += '3'
        if color_list_bands[y] == 'Yellow':
            string_first_2bands += '4'
        if color_list_bands[y] == 'Green':
            string_first_2bands += '5'
        if color_list_bands[y] == 'Blue':
            string_first_2bands += '6'
        if color_list_bands[y] == 'Purple':
            string_first_2bands += '7'
        if color_list_bands[y] == 'Gray':
            string_first_2bands += '8'
        if color_list_bands[y] == 'White':
            string_first_2bands += '9'
    bands_int = int(string_first_2bands)

    # Multiply the integer of the string with tenfold determined by color of third colorband
    if color_list_bands[2] == 'Black':
        bands_int = bands_int *  10**0
    if color_list_bands[2] == 'Brown':
        bands_int = bands_int *  10**1
    if color_list_bands[2] == 'Red':
        bands_int = bands_int *  10**2
    if color_list_bands[2] == 'Orange':
        bands_int = bands_int *  10**3
    if color_list_bands[2] == 'Yellow':
        bands_int = bands_int *  10**4
    if color_list_bands[2] == 'Green':
        bands_int = bands_int *  10**5
    if color_list_bands[2] == 'Blue':
        bands_int = bands_int *  10**6
    if color_list_bands[2] == 'Purple':
        bands_int = bands_int *  10**7
    if color_list_bands[2] == 'Gray':
        bands_int = bands_int *  10**8
    if color_list_bands[2] == 'White':
        bands_int = bands_int *  10**9

    # Make a result string of the resistorvalue
    result_str = str(bands_int) + ' OHM'

    # Print result string
    print(result_str)
    
    
    return result_str , color_list_bands


# Function to show/add the result string on our image
def show_reult(result_str, x, y, w, h, folderpath_editted_image, color_list_bands):
    print('begin showing result')

    # Read image
    img_for_showing_result = cv2.imread(folderpath_editted_image)

    # Create rectangle around detected resistor
    cv2.rectangle(img_for_showing_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put text on image
    cv2.putText(img_for_showing_result, result_str, (x, y - 5), FONT, 0.8, (255,0,255), 2, cv2.LINE_AA)
    
    # Show result image 
    cv2.imshow('Result', img_for_showing_result)
    
    # Save result image in folder 'result'
    cv2.imwrite(os.path.join(path_result, 'Image_result__' + result_str + ',' + str(color_list_bands) + ',' + 'MIN_AREA=' + str(MIN_AREA) + '.jpg'), img_for_showing_result)

    return









# Call functions
img, rectCascade = init(DEBUG, path_image)
while (not (cv2.waitKey(1) == ord('q'))):
    cliveimg = cv2.imread(path_image)
    resClose, x, y, w, h = findResistors(cliveimg, rectCascade)
    for i in range(len(resClose)):
        bandsPos_left, bandsPos_right = findBands(resClose[i], DEBUG)
        list_cluster_centers_BGR = training_clustering(path_for_training)
        color_list_bands = get_color_bands(bandsPos_left, bandsPos_right, list_cluster_centers_BGR)
        result_int, color_list_bands = calculate_result(color_list_bands)
        show_reult(result_int, x, y, w, h, path_image, color_list_bands)

    cv2.imshow("Frame", cliveimg)

cv2.waitKey(3000)
cv2.destroyAllWindows()


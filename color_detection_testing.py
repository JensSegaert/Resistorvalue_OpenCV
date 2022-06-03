import numpy
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
from PIL import Image, ImageEnhance


# Example list of left points of contours
Left = [(117, 112), (181, 46), (244, 155)]

# Example list of right points of contours
Right = [(153, 44), (213, 161), (270, 159)]

def get_color_bands(Left, Right):
    print('begin function testing')

    if len(Left) > 3:
        del Left[3:]
    if len(Left) < 3:
        print('must have 3 contours to detect resistor')
        return    # Stop function because all contours are not detected
    else: # So if there are 3 contours
        Left = Left
    if len(Right) > 3:
        del Right[3:]
    if len(Right) < 3:
        print('must have 3 contours to detect resistor')
        return    # Stop function because all contours are not detected
    else: # So if there are 3 contours
        Right= Right
    print('Left')
    print(Left)
    print('Right')
    print(Right)
    Contour_center_list = []

    # Define path for image used in testing
    path_editted_image = "C:\\Users\\Jens Segaert\\Documents\\Resistorvalue_OpenCV-main\\images_code\\editted_image.png"

    # Open the image
    img = Image.open(path_editted_image)

    # Adjust image color balance
    enhancer = ImageEnhance.Color(img)

    # Image color balance factor:
    factor = 2.5

    # Output editted input image
    editted_output = enhancer.enhance(factor)


    # Get center point of each contour by taking mean value of x and y : [(x1+x2)/2 , (y1+y2)/2]
    resistor_close_up_for_rectangles_bands = cv2.imread(path_editted_image)
    copy_resistor_clean = resistor_close_up_for_rectangles_bands.copy()
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
    for contour_center in range(len(Contour_center_list)):
        BandClose = []
        x = [round(float(s)) for s in Contour_center_list[contour_center][0]] # still a list, must be integer
        x = x.pop()
        print('x')
        print(x)
        y = [round(float(s)) for s in Contour_center_list[contour_center][1]] # still a list, must be integer
        y = y.pop()
        print('y')
        print(y)

        # Determine left, right, top and bottom for rectangle of center of resistorband
        left = x - 3
        top = y
        right = x + 3
        bottom = y - 15

        rectangle_colorband = cv2.rectangle(resistor_close_up_for_rectangles_bands, (left,bottom),(right,top),(255,255,255),1)

        # Get new image of contour center
        im1 = copy_resistor_clean[bottom: top, left: right]

        # Show image of contourcenters
        cv2.imshow('Band_contour_center', rectangle_colorband)

        # Crop center of band in form of rectangle out of image
        cv2.imread('resistor_close-up.jpg')


        # Calculate mean BGR value of contour center rectangle image
        avg_color_per_row_contourcenter = numpy.average(im1, axis=0)
        avg_color_center_contour = numpy.average(avg_color_per_row_contourcenter, axis=0)
        tuple_avg_color_center_contour = tuple(avg_color_center_contour)
        list_avg_color_contours.append(list(avg_color_center_contour))

        print('avg_color_center_contour')
        print(tuple_avg_color_center_contour)
        print(list_avg_color_contours)


    # Define cluster center list from training
    list_cluster_centers = [[ 38.22085746 , 98.11722512 , 35.76544933],
    [192.97030517 ,196.08948183 ,197.74002779],
    [ 49.94317435 , 83.70944468, 203.27440964],
    [106.20508042  ,95.75360616 ,101.46961045],
    [ 19.05106104  ,22.17084661  ,28.4488445 ],
    [ 35.25962515 ,153.31155993 ,183.01552692],
    [ 39.89947565  ,31.88809552 ,165.05747581],
    [ 76.98457262  ,46.12645161  ,58.36318146],
    [132.16414392  ,85.85733214  ,33.38264935],
    [ 27.09556064  ,33.71972816  ,84.54691794]]

    # Make list with colors
    list_colors = ['Green', 'White', 'Orange', 'Gray', 'Black', 'Yellow', 'Red',  'Purple', 'Blue', 'Brown']


    color_list_bands = []
    # Calculate norm between the centerband value and each color cluster center point (from training python files) in BGR
    # And look where minimum distance is and link this minimum to a color
    # Do this for all contour bands
    for u in range(0,len(list_avg_color_contours)):
      list_distances = []
      for w in range(0,len(list_cluster_centers)):
          from scipy.spatial import distance

          dst = distance.euclidean(tuple(list_cluster_centers[w]), tuple(list_avg_color_contours[u]))
          print('dst')
          print(dst)
          list_distances.append(dst)
          print('list distances')
          print(list_distances)
          print('min list_distances')
          print(min(list_distances))




      for l in range(0,len(list_distances)):
             print('cycle')
             if list_distances[l] == min(list_distances):
                color_list_bands.append(list_colors[l])
      print('color_list_bands')
      print(color_list_bands)

      # Plot BR cluster center- values
      for i in range(0, len(list_cluster_centers)):
            # plot B,R values of cluster centers of training list above
            plt.scatter(list_cluster_centers[i][0], list_cluster_centers[i][2])

    return color_list_bands





def calculate_result(color_list_bands):
    print('begin showing result')
    
    string_first_2bands = ''
    
    # Create integer string of first 2 bands
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

    # Multiply int with tenfold determined by color third resistorband
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
    print(str(bands_int) + ' OHM')
    return bands_int


color_list_bands = get_color_bands(Left, Right)
bands_int = calculate_result(color_list_bands)

cv2.waitKey(7000)

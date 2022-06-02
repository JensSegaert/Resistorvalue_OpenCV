Author = Jens Segaert

Machine vision project for obtaining Bachelor's degree in industrial sciences electromechanical engineering.

# Resistorvalue_Open-cv
A python script using OpenCV and haar cascades to identify and calculate the values of resistors from a single input image.


## How it works

### Haar Cascade
Using a Haar Cascade, an object classifier trained, in this case, to detect features belonging to resistors, the input is a picture of a resistor.

### Adaptive Threshold
A digital zoom is applied to each area in which a resistor was detected. The subimage is modified with a bilateral filter. An adaptive threshold is then applied which filters out the background as well as the body of the resistor itself (this minimizes the effect of the colour of the resistor). What's left from the threshold are the areas of large contrast: the coloured bands and the edges between the resistor and the background.  


### Detecting the Colours
After converting to the HSV colour space, the bilaterally filtered image is scanned for pre-defined colour ranges. These ranges may need to be tweaked depending on the white-balance of the camera in use. A mask is created for each colour that is ANDed with the adaptive threshold. The resulting mask is then filtered based on size and shape constraints to determine if the validity of this colour band. The order of the detected bands and thus the value of the resistor are then calculated. The tolerance band should be placed on the right for a 'forwards' reading. 



### Kmeans clustering
The script 'color_detection_training_general_colors.py' uses example pictures (from directory: 'Images_training_general') of which color the resistor bands could exist. These are pictures made of the color bands with a snipping app on my pc from resistorpictures from the internet and resistors at home. The mean BGR-values from each picture are devided into clusters.
In 'color_detection_testing.py' k_nearest_neighbors is used to then put the pixel values of the color bands in a certain category cluster to define the color.

### Cluster training
If you want to plot the Blue and Red values of the clusters RGB-values: 
Run one of the 'color_detection_training...'- scripts.
The one i focused on and used data from is 'color_detection_training_specific_resistorfactory'
The script 'color_detection_training_general_colors' is not trained enough (not enough pictures for training).



### Limitations
Cameras have a hard time picking up the reflective bands. Depending on the lighting, the bands may look white or even the colour of nearby objects. It is for this reason that the tolerance is not calculated.


## Get started by yourself
Python version: 3.6.8,
Install all needed packages for code.

Get input images out of directory 'input_images'. You can put your own resistor pictures in this directory as well.
You need to change paths (line 31 to line 41 in 'main.py') and the minimum area at line 28 needed for contour validation of resistorbands. 
If you want to see images of the steps on how your input image is being procesed? --> look in directory 'images_code' after running the script.
If you want to see the result of your input image? --> look in directory 'result'.
Use input images out of directory 'input_images'.
In the directory 'Output_images_code', there are results of the input images that are used by myself. In the name of
each picture, the result of the resistorvalue, colors of the 3 bands and minimum used area is given.

### Important for contour detection colorbands!
Change the parameter 'MIN_AREA' on line 28 in 'main.py'. This is de minimal area the 
contours must possess so we count it as a colorband. Sometimes with different pictures depending on the zooming,
this script can either see too many contours or too less contours.
The parameter 'MIN_AREA' is also given in the name of the resultpicture in directory 'result'.




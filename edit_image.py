#########################################################################################################################
""""
Author: Jens Segaert
Date: 03/06/2022
"""
#########################################################################################################################

# Import Libraries
import cv2
from PIL import Image, ImageEnhance
import os


# Function to give color balance to input image and show result image
def edit_input_image(path_image):
    
    # Open image
    img = Image.open(path_image)
    
    # Apply color balance on image
    enhancer = ImageEnhance.Color(img)

    # Color balance the image with factor 2.5
    factor = 2.5

    # Create output image with color balance of input image
    colored_output = enhancer.enhance(factor)

    # Show input image
    img.show()
    
    # Show output image
    colored_output.show()
    
    return



# Call function (input: given path of image that needs to be editted with color balance)
edit_input_image(os.path.abspath(os.getcwd()) + '\input_images\img_4.png')


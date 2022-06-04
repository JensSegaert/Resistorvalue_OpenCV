#########################################################################################################################
""""
Author: Jens Segaert
Date: 03/06/2022
"""
#########################################################################################################################

# Import Libraries
import cv2
from PIL import Image, ImageEnhance

# Function to give color balance to input image and show result image
def edit_input_image(path_image):
    img = Image.open(path_image)
    enhancer = ImageEnhance.Color(img)

    # Color balance the image with factor 2.5
    factor = 2.5

    colored_output = enhancer.enhance(factor)

    # Show both input image and result image
    img.show()
    colored_output.show()
    return

# Call function (input: given path of image that needs to be editted)
edit_input_image(os.path.abspath(os.getcwd()) + '\input_images\img_4.png')


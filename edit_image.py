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

edit_input_image('C:\\Users\\Jens Segaert\\Documents\\Resistorvalue_OpenCV-main\\input_images\\img.png')


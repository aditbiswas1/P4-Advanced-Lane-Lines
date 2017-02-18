import cv2

'''
read_image is a helper function to read images in openCV and tranform to RGB.
'''
def read_image(image_file):
    image = cv2.imread(image_file)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image
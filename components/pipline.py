import numpy as np
import cv2

'''
This class is 
'''
class ImagePipeline(object):


    def __init__(self):
        self.ops = []

    def add(self, op_function):
        self.ops.append(op_function)

    def apply(self, image, verbose=False, position=-1):
        if verbose:
            return
        else:
            transformed_image = image
            for op in self.ops:
                transformed_image = op(image)
            return transformed_image

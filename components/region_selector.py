import cv2
import numpy as np

'''
The RegionSelector class can be used to choose
points of interest of images taken from the front
of a car, it will find an appropriate trapezium on the 
road given a central horizon point and a top and bottom width.
These points are then used to perspective transform an image to
create a birds eye view.
'''
class RegionSelector(object):
    def __init__(self):
        self.top_left_x = 0
        self.top_left_y = 0
        self.top_right_x = 0
        self.top_right_y = 0
        self.bottom_left_x = 0
        self.bottom_left_y = 0
        self.bottom_right_x = 0
        self.bottom_right_y = 0
        self.M = None
        self.Minv = None
        self.original_image = None
        
    def show_selection(self, image):
        pts = np.array([
                [self.top_left_x, self.top_left_y],
                [self.top_right_x, self.top_right_y],
                [self.bottom_right_x, self.bottom_right_y],
                [self.bottom_left_x, self.bottom_left_y]
            ], np.int32)
        pts = pts.reshape((-1, 1 ,2 ))
        imcopy = np.copy(image)

        selected_region = cv2.polylines(imcopy, [pts], True, 255, 3)
        return selected_region
    
    
    def update_points(self,
                     bottom_y,
                     top_y,
                     center_x,
                     bottom_width,
                     top_width):
        self.top_left_y = top_y
        self.top_right_y = top_y
        self.bottom_left_y = bottom_y
        self.bottom_right_y = bottom_y
        self.top_left_x = center_x - top_width
        self.top_right_x = center_x + top_width
        self.bottom_left_x = center_x - bottom_width
        self.bottom_right_x = center_x + bottom_width
        self.M = None
        self.Minv = None
        
    def update_and_show(self,
                       bottom_y,
                       top_y,
                       center_x,
                       bottom_width,
                       top_width,
                       image):
        
        self.update_points(bottom_y,
                     top_y,
                     center_x,
                     bottom_width,
                     top_width)
        return self.show_selection(image)
        

    def warp(self,image):
        self.original_image = image
        if self.M is None:
            src = np.float32([
                [self.top_right_x, self.top_right_y],
                [self.bottom_right_x, self.bottom_right_y],
                [self.bottom_left_x, self.bottom_left_y],
                [self.top_left_x, self.top_left_y],
            ])

            dst = np.float32([
                [self.bottom_right_x, 0],
                [self.bottom_right_x,self.bottom_right_y],
                [self.bottom_left_x, self.bottom_left_y],
                [self.bottom_left_x, 0]
            ])
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.Minv = cv2.getPerspectiveTransform(dst, src)


        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def warp_inverse(self, image):
        if self.Minv is None:
            raise Exception("no Minv precalculated")
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
import cv2
import numpy as np

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
        
    def show_selection(self, image, color_tranform=True):
        pts = np.array([
                [self.top_left_x, self.top_left_y],
                [self.top_right_x, self.top_right_y],
                [self.bottom_right_x, self.bottom_right_y],
                [self.bottom_left_x, self.bottom_left_y]
            ], np.int32)
        pts = pts.reshape((-1, 1 ,2 ))
        if color_tranform:
            image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        selected_region = cv2.polylines(image, [pts], True, (255,0,0), 3)
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
        
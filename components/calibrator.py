'''
The functions of this module help calibrate images from a camera given a set of chessboard images
'''

import cv2
import numpy as np


class Caibrator(object):


    def find_chessboard_corners(self, image, nx, ny):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found_flag, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        return found_flag , coners


    def calibrate_from_chessboards(self):
        n = len(self.chessboards)
        image_points = []
        object_points = []
        img_size = None

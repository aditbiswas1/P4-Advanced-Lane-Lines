import cv2
import numpy as np


'''
The Calibrator class contains utilities to keep track of of 
the distortion matrix of a given camera.
It needs to be instanted with a set of chessboard images taken
from the camera and the number of points along a row and column of
the chessboard images
'''
class Calibrator(object):


    def find_chessboard_corners(self, image, nx, ny):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found_flag, corners = cv2.findChessboardCorners(gray_image, (nx,ny), None)
        return found_flag , corners


    def calibrate_from_chessboards(self):
        n = len(self.chessboards)
        image_points = []
        object_points = []
        img_size = None

        for image in self.chessboards:
            found, corners = self.find_chessboard_corners(image, self.nx, self.ny)
            if img_size is None:
                img_size = (image.shape[1], image.shape[0])
            if found:
                image_points.append(corners)
                objp = np.zeros((self.nx*self.ny, 3), np.float32)
                objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
                object_points.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size,None,None)
        self.mtx = mtx
        self.dist = dist

    
    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    
    def __init__(self, chessboard_images, nx=9, ny=6):
        self.chessboards = chessboard_images
        self.nx = nx
        self.ny = ny
        self.calibrate_from_chessboards()

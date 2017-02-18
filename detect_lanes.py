from components.pipeline import ImagePipeline
from components.calibrator import Calibrator
from components.region_selector import RegionSelector
from components.binarizer import ImageBinarizer
from components.utils import read_image
import cv2
import numpy as np


class LaneDetector(object):

    def __init__(self, chessboard_images):
        self.image_pipeline = ImagePipeline()
        self.calibrator = Calibrator(chessboard_images)
        self.region_selector = RegionSelector()
        self.binarizer = ImageBinarizer()

        self.image_pipeline.add(calibrator.undistort)
        self.image_pipeline.add(region_selector.warp)
        self.image_pipeline.add(binarizer.binarize)


    def detect(self, image):
        return self.image_pipeline.apply(image)




chessboard_image_sources = glob.glob('camera_cal/calibration*.jpg')
chessboard_images = [read_image(filename) for filename in chessboard_image_sources]
lane_detector = LaneDetector(chessboard_images)



from components.pipeline import ImagePipeline
from components.calibrator import Calibrator
from components.region_selector import RegionSelector
from components.binarizer import ImageBinarizer
from components.lanefinder import LaneFinder
from components.utils import read_image

import cv2
import glob
import numpy as np
from moviepy.editor import VideoFileClip


class LaneDetector(object):

    def __init__(self, chessboard_images):
        self.image_pipeline = ImagePipeline()
        self.calibrator = Calibrator(chessboard_images)

        self.region_selector = RegionSelector()
        self.region_selector.update_points(700,460,640,640,90)
        
        self.binarizer = ImageBinarizer(
                                    sobel_x_thresh=(20, 255),
                                    sobel_y_thresh=(1, 255),
                                    mag_thresh=(32, 255),
                                    dir_thresh=(0, 0.5),
                                    hls_thresh=(172, 255))

        self.lanefinder = LaneFinder(self.binarizer, self.region_selector)

        self.image_pipeline.add(self.calibrator.undistort)
        self.image_pipeline.add(self.binarizer.binarize)
        self.image_pipeline.add(self.region_selector.warp)
        self.image_pipeline.add(self.lanefinder.plot_lane_on_image)        


    def detect(self, image):
        return self.image_pipeline.apply(image)




chessboard_image_sources = glob.glob('camera_cal/calibration*.jpg')
chessboard_images = [read_image(filename) for filename in chessboard_image_sources]
lane_detector = LaneDetector(chessboard_images)


clip1 = VideoFileClip('project_video.mp4')
project_clip = clip1.fl_image(lane_detector.detect)
project_clip.write_videofile('main_project_output_2.mp4', audio=False)
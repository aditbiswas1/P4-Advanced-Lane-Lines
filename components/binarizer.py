import cv2
import numpy as np

'''
This class generates binary images from the input image
based on thresholding parameters
'''
class ImageBinarizer(object):


    def __init__(self,
        sobel_x_thresh=(0,255),
        sobel_y_thresh=(0,255),
        mag_thresh=(0,255),
        dir_thresh=(0, np.pi/2),
        hls_thresh=(0,255),
        sobel_kernel_mag=3,
        sobel_kernel_dir=15):
        self.sobel_x_thresh = sobel_x_thresh
        self.sobel_y_thresh = sobel_y_thresh
        self.mag_thresh = mag_thresh
        self.dir_thresh = dir_thresh
        self.hls_thresh = hls_thresh
        
        # helpful things for sobel thresholding
        self.sobel_kernel_mag = sobel_kernel_mag
        self.sobel_kernel_dir = sobel_kernel_dir
        self.gray_image = None
        self.sobel_x_image = None
        self.sobel_y_image = None


    def convert_to_gray(self, image):
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.gray_image

    
    def get_sobel_x(self, image):
        if self.sobel_x_image is None:
            gray = self.convert_to_gray(image)
            self.sobel_x_image = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        return self.sobel_x_image
    

    def get_sobel_y(self,image):
        if self.sobel_y_image is None:
            gray = self.convert_to_gray(image)
            self.sobel_y_image = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        return self.sobel_y_image


    def abs_sobel_thresh_x(self, image):
        sobel = self.get_sobel_x(image)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= self.sobel_x_thresh[0]) & (scaled_sobel <= self.sobel_x_thresh[1])] = 1
        return sbinary


    def abs_sobel_thresh_y(self, image):
        sobel = self.get_sobel_y(image)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sbinary = np.zeros_like(scaled_sobel)
        sbinary[(scaled_sobel >= self.sobel_y_thresh[0]) & (scaled_sobel <= self.sobel_y_thresh[1])] = 1
        return sbinary


    def mag_threshold(self, image):
        # Convert to grayscale
        gray = self.convert_to_gray(image)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_mag)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel_mag)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= self.mag_thresh[0]) & (gradmag <= self.mag_thresh[1])] = 1

        # Return the binary image
        return binary_output


    def dir_threshold(self, image):
        gray = self.convert_to_gray(image)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_dir)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel_dir)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])] = 1
        return binary_output


    def hls_select(self, image):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        binary_output[(S > self.hls_thresh[0]) & (S <= self.hls_thresh[1])] = 1
        return binary_output

        
    def binarize(self, image):
        self.original_image = image
        self.gray_image= None
        self.sobel_y_image = None
        self.sobel_x_image = None
        gradx = self.abs_sobel_thresh_x(image)
        grady = self.abs_sobel_thresh_y(image)
        grad_mag = self.mag_threshold(image)
        grad_dir = self.dir_threshold(image)
        grad_hls = self.hls_select(image)

        combined = np.zeros_like(grad_dir)
        combined[((gradx == 1) & (grady == 1)) | ((grad_mag == 1) & (grad_dir == 1)) | (grad_hls == 1)] = 1
        return combined * 255







import cv2
import numpy as np
from collections import deque
from copy import deepcopy

'''
This class determines points which represent the lane lines
from a given birds eye view image.
'''
class LaneFinder(object):

    def __init__(self, binarizer, region_selector, nwindows=9, margin=100, minpix=50, y_eval=500):
        self.nwindows =  nwindows
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix

        self.left_fit_sliding = None
        self.right_fit_sliding = None
        self.y_eval = y_eval
        self.region_selector = region_selector
        self.binarizer = binarizer

    '''
    carries out sliding window search to determine polynomials
    '''
    def sliding_window_search(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,0,255), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,0,255), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        self.left_fit_current = np.polyfit(lefty, leftx, 2)
        self.right_fit_current = np.polyfit(righty, rightx, 2)
        self.left_fit_sliding = deepcopy(self.left_fit_current)
        self.right_fit_sliding = deepcopy(self.right_fit_current)
        return self.left_fit_current, self.right_fit_current


    '''
    use sliding windows from previous iterations to determine polynomial in new frame
    '''
    def non_sliding_window_search(self, binary_warped):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin
        left_lane_inds = ((nonzerox > (self.left_fit_sliding[0]*(nonzeroy**2) + self.left_fit_sliding[1]*nonzeroy + self.left_fit_sliding[2] - margin)) & (nonzerox < (self.left_fit_sliding[0]*(nonzeroy**2) + self.left_fit_sliding[1]*nonzeroy + self.left_fit_sliding[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit_sliding[0]*(nonzeroy**2) + self.right_fit_sliding[1]*nonzeroy + self.right_fit_sliding[2] - margin)) & (nonzerox < (self.right_fit_sliding[0]*(nonzeroy**2) + self.right_fit_sliding[1]*nonzeroy + self.right_fit_sliding[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit_current = np.polyfit(lefty, leftx, 2)
        self.right_fit_current = np.polyfit(righty, rightx, 2)
        return self.left_fit_current, self.right_fit_current

    '''
    return points reperesenting the lane line on the image
    '''
    def show_line(self, binary_warped, slide_search=True):
        if (self.left_fit_sliding is  None) or (slide_search is True):
            self.sliding_window_search(binary_warped)
        else:
            self.non_sliding_window_search(binary_warped)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = self.left_fit_current[0]*ploty**2 + self.left_fit_current[1]*ploty + self.left_fit_current[2]
        right_fitx = self.right_fit_current[0]*ploty**2 + self.right_fit_current[1]*ploty + self.right_fit_current[2]
        return left_fitx, right_fitx


    '''
    estimate curvature of the road using the polynomials for the lanes
    '''
    def get_current_curvature(self):
        left_curvature = np.absolute(((1 + (2 * self.left_fit_current[0] * self.y_eval + self.left_fit_current[1])**2) ** 1.5) \
                        /(2 * self.left_fit_current[0]))
        right_curvature = np.absolute(((1 + (2 * self.right_fit_current[0] * self.y_eval + self.right_fit_current[1])**2) ** 1.5) \
                        /(2 * self.right_fit_current[0]))

        # convert to meters
        left_curvature  = left_curvature / 128 * 3.7
        right_curvature  = right_curvature / 128 * 3.7

        return (left_curvature, right_curvature)


    '''
    estimate the offset of the vehicle from the center
    '''
    def get_offset_center(self):
        left_x = self.left_fit_current[0]*self.y_eval**2 + self.left_fit_current[1]*self.y_eval + self.left_fit_current[2]
        right_x = self.right_fit_current[0]*self.y_eval**2 + self.right_fit_current[1]*self.y_eval + self.right_fit_current[2]
        center =  left_x + right_x / 2
        center_offset = (center - 640) /128 * 3.7

        return center_offset

    '''
    detect polynomials, estimate the curvature and centers and draw back the results on the 
    initial input of the pipeline
    '''
    def plot_lane_on_image(self, binary_warped):
        original_image = self.binarizer.original_image
        left_fitx , right_fitx = self.show_line(binary_warped)
        
        # curvature statistics
        current_left_curvature, current_right_curvature = self.get_current_curvature()
        curvature = (current_left_curvature + current_right_curvature) / 2
        center_offset = self.get_offset_center()


        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.region_selector.warp_inverse(color_warp) 
        # Combine the result with the original image
        image_with_lane =  cv2.addWeighted(original_image , 1, newwarp, 0.3, 0)

        # plot stats on the image
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image_with_lane, 'Radius of  Curvature = %d m' % curvature , (50, 70), font, 2, (255, 255, 255), 2)        
        cv2.putText(image_with_lane, 'Offset from Center = %.2f cm' % center_offset , (50, 130), font, 2, (255, 255, 255), 2)        
        
        return image_with_lane
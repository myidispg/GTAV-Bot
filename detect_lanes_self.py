# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:33:58 2018

@author: myidi
"""

import cv2
import numpy as np

image = cv2.imread('line_gtaV-crop.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', image)
cv2.waitKey(0)
"""
Before we detect our edges, we need to make it clear what we are exactly looking for.
Lane lines are always yellow and white. Yellow can be tricky to isolate in RGB space.
So we convert to HSV space(Hue, Saturation, Value). Next, apply a digital mask to 
return the pixels we are interested in.
We can find a target range for yellow values by a Google search.
"""
# Convert to HSV
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Target range of yellow for lanes. Based on a google search
lower_yellow = np.array([20, 100, 100], dtype='uint8')
upper_yellow = np.array([30, 255, 255], dtype='uint8')

mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow) # find only yellow
mask_white = cv2.inRange(gray_image, 200, 255) # find only white
mask_yw = cv2.bitwise_or(mask_white, mask_yellow) # merge both yellow and white
mask_yw_image = cv2.bitwise_and(gray_image, mask_yw) # use the merged ones from original img.

cv2.imshow('image', mask_yw_image)
cv2.waitKey(0)

"""
Now we will use a Gaussian filter to filter out the noise and smoothen the image.
The image will be slightly blurred but noise will be reduced.
"""

kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

cv2.imshow('window', gauss_gray)
cv2.waitKey(0)

""" Use canny edge detection to detect the edges.
 We will need to supply thresholds for canny() as it computes the gradient.
 John Canny himself recommended a low to high threshold ratio of 1:2 or 1:3.
"""
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)

cv2.imshow('window', canny_edges)
cv2.waitKey(0)

"""
We don’t want our car to be paying attention to anything on the horizon
, or even in the other lane. Our lane detection pipeline should focus on 
what’s in front of the car. Do do that, we are going to create another mask
 called our region of interest (ROI). Everything outside of the ROI will be 
 set to black/zero, so we are only working with the relevant edges.
"""

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only the keep the image in the polygon defined by the vertices.
    Everything else is set to black.
    """
    
    # Define a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

imshape = image.shape
lower_left = [imshape[1]/9,imshape[0]]
lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
roi_image = region_of_interest(canny_edges, vertices)

"""
https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
The big take away is that in XY space lines are lines and points are points,
 but in Hough space lines correspond to points in XY space and points
 correspond to lines in XY space. This is what our pipeline will look like:

Pixels are considered points in XY space
hough_lines() transforms these points into lines inside of Hough space
Wherever these lines intersect, there is a point of intersection in Hough space
The point of intersection corresponds to a line in XY space
"""

def draw_lines(img, lines, color=[255, 255, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).
    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, min_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

#rho and theta are the distance and angular resolution of the grid in Hough space
#same values as quiz
rho = 2
theta = np.pi/180
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = 20
min_line_len = 50
max_line_gap = 200

line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)

cv2.imshow('window', line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)


cv2.imshow('window', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# undistort image using camera calibration matrix from above
def undistort(img, dist, mtx):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
	
    return undist

def warpPoint(img):

    hight,width = img.shape[:2]
	
    # define source and destination points for transform
    src = np.float32([(575,464), (707,464), (258,682), (1049,682)])	
    dst = np.float32([(450,0), (width-450,0), (450,hight), (width-450,hight)])
	
    return src, dst, hight, width
	
def unwarp(img):

    src, dst, h, w = warpPoint(img)
	
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
	
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    binary_output = np.zeros_like(scaled_sobel)
            # is > thresh_min and < thresh_max
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1            
    # 6) Return this mask as your binary_output image
    
    return binary_output
	
def displayResult( img1, img2, title1, title2 ):

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	fig.subplots_adjust(hspace = .2, wspace=.05)

	ax1.imshow(img1)
	ax1.set_title(title1, fontsize=20)

	ax2.imshow(img2)
	ax2.set_title(title2, fontsize=20)
	

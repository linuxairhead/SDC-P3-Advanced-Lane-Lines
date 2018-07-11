# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Chessboard.png 
[image2]: ./output_images/UndistortedImage.png
[image3]: ./output_images/UnwarpedImage.png 
[image4]: ./output_images/XSobelAbsolute.png 
[image5]: ./output_images/YSobelAbsolute.png 
[image6]: ./output_images/GradientMagnitude.png 
[image7]: ./output_images/GradientDirection.png 
[image8]: ./output_images/GradientCombined.png 
[image9]: ./output_images/thresholdedS.png 
[image11]: ./output_images/pipelineResultForSix1.png
[image12]: ./output_images/pipelineResultForSix2.png 
[image13]: ./output_images/pipelineResultForSix3.png 
[image14]: ./output_images/pipelineResultForSix4.png 
[image15]: ./output_images/pipelineResultForSix5.png 
[image16]: ./output_images/pipelineResultForSix6.png 
[image17]: ./output_images/pipelineResultForSix7.png 
[image18]: ./output_images/pipelineResultForSix8.png 
[image19]: ./output_images/PeaksHistogram.png 
[image20]: ./output_images/slidingWindows.png 
[image21]: ./output_images/windowMargin.png 
[image22]: ./output_images/CombineResult.png 
[image23]: ./output_images/CombineResult2.png 
[image24]: ./output_images/CombineResult3.png 
[image25]: ./output_images/CombineResult4.png 
[image26]: ./output_images/CombineResult5.png 
[image27]: ./output_images/CombineResult6.png 
[image28]: ./output_images/CombineResult7.png 
[image29]: ./output_images/CombineResult8.png
[image30]: ./output_images/CombineResultWText.png
[image31]: ./output_images/CombineResultWText2.png
[image32]: ./output_images/CombineResultWText3.png
[image33]: ./output_images/CombineResultWText4.png
[image34]: ./output_images/CombineResultWText5.png
[image35]: ./output_images/CombineResultWText6.png
[image36]: ./output_images/CombineResultWText7.png
[image37]: ./output_images/CombineResultWText8.png

[video1]: ./project_video_output.mp4 

## Camera Calibration & image Transform

### 1. Computed coefficients

		# Make a list of calibration images
		images = glob.glob('./camera_cal/calibration*.jpg')

		fig, axs = plt.subplots(4,5, figsize=(16, 11))
		fig.subplots_adjust(hspace = .2, wspace=.05)
		axs = axs.ravel()

		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# Step through the list and search for chessboard corners
		for i, fname in enumerate(images):
			img = cv2.imread(fname)
			grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				
			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(grayImg, (column,row), None)

			# If found, add object points, image points
			if ret == True:
				objpoints.append(objp)
				
				corners2 = cv2.cornerSubPix(grayImg, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners2)
			
				# Draw and display the corners
				img = cv2.drawChessboardCorners(img, (column, row), corners, ret)
				axs[i].axis('off')
				axs[i].imshow(img)
		

* Preparing "object points", which pick as (x, y, z) coordinates of the chessboard corners in the world. Since the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  And `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it, once detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

		# Test undistortion on an image
		img = cv2.imread('./camera_cal/calibration02.jpg')
		img_size = (img.shape[1], img.shape[0])

		# Do camera calibration given object points and image points
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
		dst = cv2.undistort(img, mtx, dist, None, mtx)

* Using the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  By applying this distortion correction to the test image using the `cv2.undistort()` function, able to return following image. 

![alt text][image1]


### 2. Corrected image using computed coeffieient.

		# Choose a test image 
		aImg = cv2.imread('./test_images/straight_lines1.jpg')
		aImg = cv2.cvtColor(aImg, cv2.COLOR_BGR2RGB)

		aImg_Undistort = cv2.undistort(img, mtx, dist, None, mtx)

* Again Using computed coefficients, I was able to corrected from the camera distortion image from test image as following.

![alt text][image2]

### 3. Perspective Transform

		def warpPoint(img):

			hight,width = img.shape[:2]
			
			# define source and destination points for transform
			src = np.float32([(570,460), (710,460), (205,700), (1100,700)])	
			dst = np.float32([(200,0), (width-200,0), (200,hight), (width-200,hight)])
			
			return src, dst, hight, width
			
		def unwarp(img):

			src, dst, h, w = warpPoint(img)
			
			# use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
			M = cv2.getPerspectiveTransform(src, dst)
			Minv = cv2.getPerspectiveTransform(dst, src)
			
			# use cv2.warpPerspective() to warp your image to a top-down view
			warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
			return warped, M, Minv
			
		aImg_unwarp, M, Minv = unwarp(aImg_Undistort)	

* Since objects appear smaller as farther away, it was hard to identify the curve from the image, so I used perspective transfrom to I transform selected road to bird's eye view of image. 

* Transform  point

	|    Source     |  Destination      | 
	|:-------------:|:-----------------:| 
	|   570, 460    |     200, 0        | 
	|   710, 460    |   width-200, 720  |
	|   205, 700    |    200, hight     |
	|  1100, 700    |  width-200, hight |

* Once image was transfrom, it was easier to identify the curve as you could see from following.
 		
![alt text][image3]

---

## Canny edge detection algorithm

### 1. Applying Sobe Operator

		def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
			
			# Apply the following steps to img
			# 1) Convert to grayscale
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
			
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

* the Sobel operator to an image is a way of taking the derivative of the image in the xx or yy direction.
* As you can see from below result, the image from xx direction of sobel operator gave me emphasized edge closer to vertical, which result was more usefull then the image from yy direction of sobel operator.   

![alt text][image4]
![alt text][image5]

### 2. Magnitude of the Gradient 

		# Define a function that applies Sobel x and y, 
		# then computes the magnitude of the gradient
		# and applies a threshold
		def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
			
			# Apply the following steps to img
			# 1) Convert to grayscale
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# 2) Take the gradient in x and y separately
			sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
			sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
			# 3) Calculate the magnitude 
			gradmag = np.sqrt(np.square(sobelx) + np.square(sobely))
			# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
			scale_factor = np.max(gradmag)/255 
			gradmag = (gradmag/scale_factor).astype(np.uint8)     
			# 5) Create a binary mask where mag thresholds are met
			binary_output = np.zeros_like(gradmag)
			binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1    
			# 6) Return this mask as your binary_output image

			return binary_output

* The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients.

![alt text][image6]

### 3. Direction of the Gradient

		# Define a function that applies Sobel x and y, 
		# then computes the direction of the gradient
		# and applies a threshold.
		def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
			# Grayscale
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# Calculate the x and y gradients
			sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
			sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
			# Take the absolute value of the gradient direction, 
			# apply a threshold, and create a binary image result
			absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
			binary_output =  np.zeros_like(absgraddir)
			binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

			# Return the binary image
			return binary_output
	
* The direction of the gradient is the inverse tangent (arctangent) of the yy gradient divided by the xx gradient:
* Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians (covering a range of −π/2 to π/2.) An orientation of 0 implies a vertical line and orientations of {+/-} π/2 imply horizontal lines

![alt text][image7]

### 3. Combine Magnitude and Direction of the Gradient

		# Apply each of the thresholding functions
		gradx = abs_sobel_thresh(aImg_unwarp, orient='x', thresh_min=20, thresh_max=100)
		mag_binary = mag_thresh(aImg_unwarp, sobel_kernel=25, mag_thresh=(20, 100))
		dir_binary = dir_threshold(aImg_unwarp, sobel_kernel=15, thresh=(0.7, 1.3))

		combined = np.zeros_like(dir_binary)
		#combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
		combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

![alt text][image8]

### 4. Combine with HLS Thresholds

		# Edit this function to create your own pipeline.
		def pipelineTest1(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
			img = np.copy(img)

			# Convert to HLS color space and separate the V channel
			hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			l_channel = hls[:,:,1]
			s_channel = hls[:,:,2]
			
			# Sobel x
			sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
			abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
			scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
			
			# Threshold x gradient
			sxbinary = np.zeros_like(scaled_sobel)
			sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
			
			# Threshold color channel
			s_binary = np.zeros_like(s_channel)
			s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
			# Stack each channel
			color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
			
			return color_binary
			
* Once take HLS (Hue, Lightness and Saturation) Color transform, I use Saturation which is measurement of the colorfulness or intentness of color. This will emphasize the lince more clearly.
  
![alt text][image9]

### 5. Pipeline Test
* Perspective Transform
* X Sober Operator
* Magnitude & Direction of the Gradient
* S thresholds

		def pipelineTest(img):
			
			# unwarp the image
			undistort_Img = undistort(img, dist, mtx)
			unwarp_Img, M, Minv = unwarp(undistort_Img)

			# Apply each of the thresholding functions
			aImg_Sobel = abs_sobel_thresh(unwarp_Img, orient='x', thresh_min=20, thresh_max=150)
			aImg_MagBinary = mag_thresh(unwarp_Img, sobel_kernel=25, mag_thresh=(30, 200))
			dir_binary = dir_threshold(unwarp_Img, sobel_kernel=25, thresh=(0.7, 1.3))
			aImg_HLS = hls_select(unwarp_Img, thresh=(170, 255))

			combined = np.zeros_like(dir_binary)
			combined[(aImg_Sobel == 1) | ((aImg_MagBinary == 1) & (dir_binary == 1) | (aImg_HLS == 1))] = 1

			return combined
	
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]

---
## Define conversions in x and y from pixels space to meters

* decide explicitly which pixels are part of the lines
	* which belong to the left line
	* which belong to the right line.
	
		histogram = np.sum(result[result.shape[0]//2:,:], axis=0)
		plt.plot(histogram)
		plt.title('Peaks in a Histogram')

![alt text][image19]

* Using sliding windows through a warped binary image and find "hot" pixels are associated with the lane lines.
	
		def sliding_windows( binary_warped ):

			# Assuming you have created a warped binary image called "binary_warped"
			# Take a histogram of the bottom half of the image
			histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
			
			# Create an output image to draw on and  visualize the result
			out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

			# Find the peak of the left and right halves of the histogram
			# These will be the starting point for the left and right lines
			midpoint = np.int(histogram.shape[0]/2)
			leftx_base = np.argmax(histogram[:midpoint])
			rightx_base = np.argmax(histogram[midpoint:]) + midpoint

			# Set height of windows
			window_height = np.int(binary_warped.shape[0]/nwindows)

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
			for window in range(nwindows):
				
				# Identify window boundaries in x and y (and right and left)
				win_y_low = binary_warped.shape[0] - (window+1)*window_height
				win_y_high = binary_warped.shape[0] - window*window_height
				win_xleft_low = leftx_current - margin
				win_xleft_high = leftx_current + margin
				win_xright_low = rightx_current - margin
				win_xright_high = rightx_current + margin    
			
				# Draw the windows on the visualization image    
				cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
				cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
				
				# Identify the nonzero pixels in x and y within the window
				good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
						(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
				good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
						(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
				
				# Append these indices to the lists
				left_lane_inds.append(good_left_inds)
				right_lane_inds.append(good_right_inds)
				
				# If you found > minpix pixels, recenter next window on their mean position
				if len(good_left_inds) > minpix:
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				if len(good_right_inds) > minpix:        
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
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)
			
			# Fit a second order polynomial to each
			left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
			right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
			
			return left_lane_inds, right_lane_inds, left_fit, right_fit, left_fit_m, right_fit_m, out_img
	
![alt text][image20]

* define the meters per pixel from the plot
	* Given condition - projecting a section of lane similar to the images
		* the lane is about 30 meters long and 3.7 meters wide.
	* Get the pixel from the image 720 as y, 935 as x 
		
			Define conversions in x and y from pixels space to meters
			ym_per_pix = 30/720 # meters per pixel in y dimension
			xm_per_pix = 3.7/935 # meters per pixel in x dimension
		
![alt text][image21]

## Measuring Curvature

* Calculate the curvature of a lane line
	* to fit a 2nd degree polynomial to that line
	* from the line, calculate the radius of curvature
	
			def calculatePixelCurvature( left_fit, right_fit, ploty ) :
				# Define y-value where we want radius of curvature
				# I'll choose the maximum y-value, corresponding to the bottom of the image
				y_eval = np.max(ploty)
				left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
				right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
				
				# Example values: 1926.74 1908.48
				return left_curverad, right_curverad			


* Calculated the new radius of curvature.
			
		def calculateActualCurvature( left_fitx, right_fitx, ploty ) :

			y_eval = np.max(ploty)
			
			# Fit new polynomials to x,y in world space
			left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
			right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
			
			# Calculate the new radii of curvature
			left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
			right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

			# Now our radius of curvature is in meters
			return left_curverad, right_curverad
			# Example values: 632.1 m    626.2 m			

---
## Combine output 

### 1. Display Lane Boundary

		def drawLine(img, left_fit, right_fit, ploty):   
		 
			color_warp = np.zeros_like(img).astype(np.uint8)
			
			# Calculate points.
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
			
			# Recast the x and y points into usable format for cv2.fillPoly()
			pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
			pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
			pts = np.hstack((pts_left, pts_right))
			
			# Draw the lane onto the warped blank image
			cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
			
			# Warp the blank back to original image space using inverse perspective matrix (Minv)
			newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
			return cv2.addWeighted(img, 1, newwarp, 0.3, 0), left_fitx, right_fitx
			
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]
![alt text][image29]

### 2. Display Curvature

		def displayCurvature(img, left_fit_m, right_fit_m, left_fitx , right_fitx, ploty ):
			fontScale=1
		 
			left_curverad, right_curverad = calculateActualCurvature( left_fitx , right_fitx, ploty ) 
			
			# Calculate vehicle center
			xMax = img.shape[1]*xm_per_pix
			yMax = img.shape[0]*ym_per_pix
			
			vehicleCenter = xMax / 2
			
			lineLeft = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
			lineRight = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
			lineMiddle = lineLeft + (lineRight - lineLeft)/2
			diffFromVehicle = (lineMiddle - vehicleCenter)
			
			if diffFromVehicle > 0:
				message = '{:.2f} m right'.format(diffFromVehicle)
			else:
				message = '{:.2f} m left'.format(-diffFromVehicle)
			
			# Draw info
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontColor = (255, 255, 255)    
			cv2.putText(img, 'Left curvature:   {:.0f} m'.format(left_curverad), (30, 50), font, fontScale, fontColor, 2)
			cv2.putText(img, 'Right curvature: {:.0f} m'.format(right_curverad), (30, 100), font, fontScale, fontColor, 2)
			cv2.putText(img, 'Vehicle is {} of center'.format(message), (30, 150), font, fontScale, fontColor, 2)    
			return img
			
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]
![alt text][image37]	
	
## Process Pipeline 

		def process_image(image):    

			ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
			pipeResult, sxbinary = pipelineTest2(image)  
		  
			left_lane_inds, right_lane_inds, left_fit, right_fit, left_fit_m, right_fit_m, out_img = sliding_windows(sxbinary)
			lineImage, left_fitx, right_fitx = drawLine(image, left_fit, right_fit, ploty)
			textLineImage = displayCurvature(lineImage, left_fit_m, right_fit_m, left_fitx , right_fitx, ploty )
			
			return textLineImage

			
		video_output = 'project_video_output.mp4'
		video_input = VideoFileClip('project_video.mp4')
		processed_video = video_input.fl_image(process_image)
		%time processed_video.write_videofile(video_output, audio=False)

		video_input.reader.close()
		video_input.audio.reader.close_proc()
	
---

## Video Result

|    Video Source            |              Video Result            | 
|:--------------------------:|:------------------------------------:| 
|       Project Video        |         [Project Video Result](./project_video_output.mp4)       |
|      Challenge Video       |        [Challenge Video Result](./project_video_output.mp4)      |
|   Harder Challenge Video   |    [Harder Challenge Video Result](./project_video_output.mp4)    |

	
---

## Discussion
* The Color difference of the road due to shadow or fade of road.
	* I was able to limited overcome by morphological transformation and Saturation Color Space from HLS. 
	* When there was various color difference, it wasn't it failed to detect the road.

* When there was curve ( harder challenge video ),  unabled to detect the lane correctly.
	* when there was a sudden curve, it failed to detect the curve correctly.
	
* I estimated the meter per pixel from line detection plot, so It wasn't always gave me correct calculate value.
	* I believe radair would give me more accurated value. 
	




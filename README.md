# CarND_project4
Advanced Lane Lines
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./report/Corners.png "Corners"
[image2]: ./report/Undistort.png "Undistort Squares"
[image3]: ./report/test1.jpg "Original Road Image"
[image4]: ./report/test1_undist.jpg "Undistored Road Image"
[image5]: ./report/test1_xgrad_magnitude.png "soebel x magnitude"
[image6]: ./report/test1_xgrad_binary.png "soebel x binary"
[image7]: ./report/test1_s_magnitude.png "S channel magnitude"
[image8]: ./report/test1_s_binary.png "S channel binary"
[image9]: ./report/test1_combined_binary.png "Combined gradient and color binary" 
[image10]: ./report/straight_lines1.png "Warped straight lines 1"
[image11]: ./report/straight_lines2.png "Warped straight lines 2"
[image12]: ./report/test1_transformed.png "Transformed"
[image13]: ./report/test1_fit.png "Fit"
[image14]: ./report/test1_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 2. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/project4.ipynb"). 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  An example of this is given below:

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I added dots at the corners, applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 3. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

The undistortion step yields:
![alt text][image4]

#### 4. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.  First, I perfromed a sobel gradient transform in the x-direction.  The x-direction was selected because lane lines tend to be more vertical than horizontal and this help filter extranious images.  The Kernel size used was 3.  The magnitute of the threshold was 35. This transform is perfomed in the in the magx__threshold function located in the cell uder the heading "2. Apply sobel gradient transform". Here are examples of the magnitude and the binary output from this step.


![alt text][image5]
![alt text][image6]

The S-channel data was used to extract lines, especially the yellow lines. The threshold was set to 200. The color thransform is applied in the x_threshod function below the "3. Apply Color Transform" heading.  Here are examples of the magnitudes and binary output from this step.

![alt text][image7]
![alt text][image8]

These transforms were combined. Below is an example resulting from the combination of gradient and color transforms.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the cell under the heading "5. Perform perspective transform".  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
fractx0=.15
fractx1=.435
fracty = 0.66
src = np.float32(
    [[img_size[0] * fractx1,       img_size[1] * fracty],
     [ img_size[0] * fractx0,       img_size[1]         ],
     [ img_size[0] * (1 - fractx0), img_size[1]         ],
     [ img_size[0] * (1 - fractx1), img_size[1] * fracty]])
dst = np.float32(
    [[(img_size[0] /4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 556, 475      | 320, 0        | 
| 192, 720      | 320, 720      |
| 1088, 720    | 960, 720      |
| 723, 475      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped images shown below.

![alt text][image10]
![alt text][image11]

An example of the transformation from an imge to a birds-eye view of binary pixels is given below.

![alt text][image12]

#### 6. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

As described in lecture 33, to find the lane lines, I performed the following steps if the fit was unknown:
1.  Transformed the image
2.  Divided the image into four quadrents
3.  For each of the bottom two quadrents, I perfomed a histogram showing the number of pixels as a function of the x positions. The maximum value in the lower left (right) quadrant was used as the starting point for the left (right)  lane line. 
4.  I used nine windows and set the margin to 100.  If at least 50 pixels were found, I recentered the window. 
5.  Once the points within the left and right windows were found, I used the below statements to fit a second order polynomial:

	    left_fit = np.polyfit(lefty, leftx, 2)
	    right_fit = np.polyfit(righty, rightx, 2) 

This was first done in the cell under the "WIndow from lecture 33" heading.  A result is given below.

![alt text][image13]

If the fit was known, the fit set the staring point and the allowable margin set the search area. 

#### 7a. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The bulk of the work is performed in the cells under the "Implement video" heading.  In the process_image() function, the below lines of code reside. 


	    # Define conversions in x and y from pixels space to meters
	    ym_per_pix = 30/720 # meters per pixel in y dimension
	    xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	    # Fit new polynomials to x,y in world space
	    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	    
	    # Calculate the new radii of curvature 
	    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
	    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) /(2*right_fit_cr[0])
	    
	    curverad = 2/(1/left_curverad + 1/right_curverad)
	    
	    left_angle_top  = np.arctan(left_fit_cr[1])  * 180 
	    right_angle_top = np.arctan(right_fit_cr[1]) * 180
	    
	    left_angle  = np.arctan(2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])  * 180 
	    right_angle = np.arctan(2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1]) * 180
	    
	    
	    # find offset
	    lx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
	    rx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
	    center_x = (lx + rx)/2.0
	    offset_pixel = center_x - 1279/2.0
	    offset = offset_pixel * xm_per_pix
	    lane_width = (rx - lx) * xm_per_pix


1. A pixel to real world approximation was made. 
2. A real world fit was performed
3. The curve radii were obtained.
4. The curvatures were averaged.
5. The inverse of the average curvature was give.  The curvatures were averaged because if a line is straigtht, or nearly straight, that line would dominate as the radius could approach infinity.
6. Angles of the lines at the top and bottom were obtained.  These will be used to make sure the lines are parallel. 
7. The offset is calculated by multiplying the real world distance to pixel distance facor and the offset between the middle of the lane and the center of the image.

#### 7b.  Describe the sanity checks you performed

Sanity checks included:
1. Making sure each radius of curvatures was greater than 10 m.
2. Making sure the offset was less than 1.5m.
3. Making sure the lane width was between 2 m and 4 m.
4. Making sure the lines were parallel within 5 degrees at both the top and bottom of the transformed image.
5. Making sure at least 200 pixels were detected for both the left lane line and the right lane line.

#### 7c.  Describe the implemtation of a low-pass filter

An exponentially weighted moving average (EWMA) filter with a smoothing constant of 0.05 was implemented.

If an image passed the sanity check:
1. Left and right curvatures (inverse of radius of curvatures) were adjuested and then averaged together.
2. Each set of baseline points was adjusted.
3. From the adjusted baseline points, the offset and lane width were calculated.
4.  From the adjusted baseline points, the angles of the left and right lanes were calculated and then averaged to obtain the yaw.


#### 8. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


The below values were selected to be shown:
1. Signed curvature.  Negative values indicted the road curves to the left.  Positive values indicate the road curves to the right. Curvature is the inverse of the radius of curvature.   This value is listed as "Curvature" in the video.
2. Angle of approach, yaw. Negative values indicted the vehicle is turned to the left.  Positive values indicate the vehicle is turned toward the right. This value is listed as "Angle" in the video.
3. Offset. Distance from the center of the lane.
4. Lane width. Width of the lane.

I implemented this step in the 'process__image()' in the function in the third to last cell of the Jupyter notebook. Below is an example output of this function.

![alt text][image14]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Click the below image for a link to my [!["video output"](https://www.youtube.com/watch?v=xcQWSDuIkVY "Video output")

[![Video output](https://img.youtube.com/vi/xcQWSDuIkVY/0.jpg)](https://www.youtube.com/watch?v=xcQWSDuIkVY "Video output")

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

By far the most difficult issues was dealing with very bright images and shadows. The recommended next step is to implement a pre-processing step to perform local normalization and/or a color space transformation and filter.  Also, I recommend trying a varying height window to make sure a second order fit is reasonable. For slow sweeping turns with great light light, a tall window should be used.  For quick turns or if lighting is poor, a short window should be used.

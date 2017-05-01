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

[calibration]: ./calibration_result.png "Undistorted"
[original]: ./test_images/test1.jpg "Road Transformed"
[threshold]: ./threshold_result.png "Binary Example"
[warp]: ./warp_result.png "Warp Example"
[line]: ./line_result.png "Fit Visual"
[final]: ./final_result.png "Output"
[video1]: ./result_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the **1st code cell** of the IPython notebook located in "./Advanced_Lane_Lines.ipynb" (or in lines # through # of the file called `some_file.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][calibration]

A few camera calibration images in `camera_cal/` don't contain full 9x6 corner points. These images are excluded in calibration processing.

### **Pipeline (single images)**

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][original]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at **2nd through 15th code cell** in `Advanced_Lane_Lines.ipynb`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][threshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 16th through **18th code cell** in the file `Advanced_Lane_Lines.ipynb`.  The `unwarp_trim()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
W, H = 1500, 1280
trim_w, trim_h = 1500, 1000
src = np.array([[594, 450], [684, 450], [1056, 690], [250,690]], np.float32)
dst = np.array([[250, 0], [1056, 0], [1056, H], [250, H]], np.float32)

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def unwarp_trim(img):
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    #delete the next two lines
    return warped[H-trim_h:]

def recover(img):
    h, w = img.shape[:2]
    new = np.zeros((H, W, 3), np.uint8)
    new[H-h:] = img
    return new
```

This resulted in the following source and destination points:

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][line]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in **19th code cell** in `Advanced_Lane_Lines.ipynb`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][final]

---

### **Pipeline (video)**

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result_project_video.mp4)

---
#### 2. Apply Chanllenge!!!

Here's a [link to my challenge video result](./result_challenge_video.mp4)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

There are many difficulties when I apply the pipeline above on challenge video. I need to upgrade lane-finding process.

First, all parameters are fine-tuned again. Simple noise filter is also added by `cv2.dilate` in HLS theresholding.

Second, if a polynomial fit was found to be robust in the previous frame, then rather than search the entire next frame for the lines, just a window around the previous detection could be searched.

Third, low pass filter added for deciding final lane lines. (in **19th code cell**). By keeping latest 5 results, and averaging them, jitter is removed.

Fourth, I add outlier checker mechanisim in lane-finding process. (in **19th code cell**) Whenever fitting lines, compare the current result with the previous result. If the difference is too big, abandon the current result as a outlier.


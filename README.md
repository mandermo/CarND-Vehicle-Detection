## Vehicle Detection project
### My solution to the "Vehicle Detection" project in the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). Here you can find a [link to Udacity's upstream project](https://github.com/udacity/CarND-Vehicle-Detection).

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/hogsearch.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[heatmap]: ./output_images/heatmap.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First I collected the paths for the training and the test images and collected them in the variables car_paths_train, car_paths_test, noncar_paths_train and noncar_paths_test. I allocated 20% of the images to the test set controlled by the variable test_size_ratio. For the GTI images I sorted them by path name and reserved the first 20% to the test set. Since many of the images are a set of similar images taken at roughly the same time, this way I get less occurances when one images appears in the test set and also in the training set. This way I hope to decrease the risk of overfitting to the test set. The code for that is under heading "Accuire Vehicle and Non-vehicle paths and split to train and test". I read in the images into memory at "Load in training and test images from the paths".

I have code for feature extraction under "Feature extraction code", which is mostly taken from the course material.

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with different parameters, but I also thought it could be smart to let GridSearchCV help in searching for the feature hyper parameters and not just search for the SVM hyperparameters. I did that by implementing my own transformer by inheriting from BaseEstimator and TransformerMixin and then implementing a no-op fit and a transform member function. Then I put my object inside a sklearn Pipeline object, together with a StandardScaler and a LinearSVC. After grid search is finished, I extract the trained StandardScalar and the SVM and also extract the found hyper parameters for the feature extraction. The code is under "Classifier Training". This was not a hit and a run of more than 8000 seconds of grid search gave worse test accuracy than what I tried by hand. I think that the reason is that the default cross validation picks samples randomly from the training set for validation, while a similar approach to pick a validation set as how I picked a test test would be better (that approach described under 1. in this document). So to summarise, I think that the the grid search find the parameters that overfit on the training set, because the validation set is too similar to the training set. Therefore when we evalutate the accuracy on the test set, the score gets low.

The parameters settled for are HLS color space, 10 hog orientations, 8 pixels per cell, 2 cells per block, and 32 spatial size and 32 hist bins. The parameters suggested by the grid search are: 18 hog orientations, 16 pixels per cell, 2 cells per block and no color histogram and no spatial bins. The grid search prefered 0.2 as C for the SVM over 0.5. I tried some other values in other grid search runs. I picked 0.2 as the C value.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using an sklearn Pipeline and as described above. The code is under "Classifier Training". I tried earlier using SVC(probability=True) and using predict_proba(), but it took longer time to train and didn't give better found bounding boxes. Now I use decision_function() with a threshold to limit false positives.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I am using the sliding window search from "Hog Sub-sampling Window Search", and call it with multiple scales. Each scale have individual ystart and ystop values. I tweaked each scale separately on the test images and then combined them all concatenating the found bounding boxes and the tweaked a bit more.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/processed_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I made a heatmap for each frame and kept thresholded keeping the values higher than 1.

![alt text][heatmap]

For the video I clipped the heatmap value for each frame to a maximum value of 1. Then I kept track of the last 7 frames and the summed then together keeping values higher than 5.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had trouble that the grid search didn't work very well to search for the hyper parameters, I would consider improving on how it selected a validation set.

The recognition was quite sensitive to seleciton of hyper parameters. I would explore some feature detection and matching method using OpenCV to track the vehicles betwen frames. By tracking feature points on the vehicle and assume that a homography could be good approximation of the transformation for feature points on a car from one frame to another frame close in time after. Using a RANSAC or similar robust estimator to reject outliers. Maybe some parameters for outlier rejection would need to be tweaked to make the homography approximation viable. Filtering of feature point on the road could probably be done by first transforming the image to birds eye view and use a feature matcher and then see if the points move at the driving speed.
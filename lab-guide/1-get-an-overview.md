# Step 1: Get an overview
We will start by presenting an overview of the method and the contents of this project.

## Algorithm overview
We will use `CornerDetector` to compute and return a list of detected keypoints. 
The main steps in the detection algorithm are:
- Compute the image gradients using derivative of Gaussian filters.
- Use the image gradients to compute the images *A*, *B*, *C* from the lectures:
  - Element-wise products of the gradients.
  - Apply windowing by convolving these with a (bigger) Gaussian.
- Use *A*, *B* and *C* to compute a corner metric for the entire image:
  - *&lambda;*<sub>min</sub>, Harris and Harmonic Mean.
- Threshold the corner metric image and find local maxima:
  - Morphological operations.
  - Logical operations.
  
We will then try to find a circle hidden in the list of detected corner keypoints. 
To achieve this, we will use `CircleEstimator` to apply RANSAC, and extract the circle estimate based on the largest set of inliers.

## Introduction to the project source files
We have chosen to distribute the code on the following modules:
- [**lab_corners.py**](../lab_corners.py)
  
  Contains the main loop of the program and all exercises. 
  Your task will be to finish the code in this module. 
  
- [**common_lab_utils.py**](../common_lab_utils.py)

  This module contains utility functions and classes that we will use both in the lab and in the solution.
  Please take a quick look through the code.
 
- [**solution_corners.py**](../solution_corners.py)

  This is our proposed solution to the lab.
  Please try to solve the lab with help from others instead of just jumping straight to the solution ;)

## `run_corners_lab()`
First, take a look at the `run_corners_lab()` function in [lab_corners.py](../lab_corners.py). 
Try to understand the steps taken here, and please ask one of the instructors if you are uncertain.

Then run the project. 
You should be able to see a video stream from the camera, but the program doesn't do anything interesting - yet!

Please continue to the [next step](2-implement-a-corner-feature-detector.md) to get started with the interesting stuff!

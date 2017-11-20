
'''
Brandon Johnson
Rollins ID: bjohnson1
“On my honor, I have not given, nor received, nor witnessed any unauthorized
 assistance on this work.”

Collaboration Statement:

I reffered to:
https://docs.opencv.org/2.4/modules/imgproc/doc/
feature_detection.html?highlight=canny
'''

import cv2
import numpy as np
import scipy as sp

""" Assignment 3 - Detecting Gradients / Edges (Manually)
"""

def image_gradient_x(image):
    """ This function differentiates an image in the X direction

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction.
    """
    # WRITE YOUR CODE HERE.
    rows, cols = image.shape
    image_copy = image.copy()

    for r in range(rows):
        for c in range(cols - 1):

            #subtract by on column from previous to find derivative in X
            image_copy[r][c] = abs( int(image_copy[r][c + 1]) - int(image_copy[r][c]) )

    return image_copy

    # END OF FUNCTION.

def image_gradient_y(image):
    """ This function differentiates an image in the Y direction.
    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction.
    """
    # WRITE YOUR CODE HERE.

    rows, cols = image.shape
    image_copy = image.copy()

    for r in range(rows-1):
        for c in range(cols):

            #subtract by on row from previous to find derivative in Y
            image_copy[r][c] = abs( int(image_copy[r+1][c]) - int(image_copy[r][c]))

    return image_copy

    # END OF FUNCTION.

def compute_gradient(image, kernel):

    """ This function applies an input 3x3 kernel to the image, and returns the
    result. This is the first step in edge detection which we discussed in
    lecture.

    You may assume the kernel is always a 3 x 3 matrix.
    View lectures on cross correlation/convolution, gradients, and edges
    to review this concept.

    The process is this: At each pixel, perform cross-correlation using the
    given kernel. Do this for every pixel, and return the output image.

    A common question for this assignment is, What do you do at image[i,j]
    when the kernel goes outside the bounds of the image? You are allowed
    to start iterating the image at image[1, 1] (instead of 0, 0) and end
    iterating at the width - 1, and column - 1.  In other words, a kernel
    of size n will create an n//2 pixel border which you are allowed to
    ignore.

   Perform a manual cross-correlation (using many nested
	for loops!). Basically, you are implementing the filter2D function (very slowly!)

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The computed gradient for the input image.
    """
    # WRITE YOUR CODE HERE.

    rows, cols = image.shape
    image_copy = image.copy()

    #Travers through image using for loop

    sum = 0
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            sum = 0

            # traverse through kernel and find summation

            for ker_r in range(-1,2):
                for ker_c in range(-1,2):

                    sum += (image_copy[r + ker_r][c +ker_c]) * (kernel[ker_r + 1][ker_c + 1])

            image_copy[r][c] = sum

    return image_copy



    # END OF FUNCTION.

"""
def convert_to_bw(image):

    rows, cols = image.shape

    for r in range(rows):
        for c in range(cols):
            #identifies pixels with value greater than 128
            #converts those pixels to 255
             if 70 < image[r][c] < 150 :
                 image[r][c] = 255

             else:
                 image[r][c] = 0

    return image

def edge(image, kernel):
    image_c = image.copy()
    image_c = compute_gradient(image_c, kernel)

    image_e = convert_to_bw(image_c)

    return image_e
"""

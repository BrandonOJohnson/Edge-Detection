import numpy as np
import cv2
import unittest

from edge_detection import image_gradient_x
from edge_detection import image_gradient_y
from edge_detection import compute_gradient

class Assignment5Test(unittest.TestCase):
    def setUp(self):
        self.testImage = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)

        if self.testImage is None:
            raise IOError("Error, image test_image.jpg not found.")

    def test_image_gradient_x(self):
        self.assertEqual(type(image_gradient_x(self.testImage)),
                         type(self.testImage))
        print("\n\nSUCCESS: image_gradient_x returns the correct output type.\n")

    def test_image_gradient_y(self):
        self.assertEqual(type(image_gradient_y(self.testImage)),
                         type(self.testImage))
        print("\n\nSUCCESS: image_gradient_y returns the correct output type.\n")

    def test_compute_gradient(self):
        avg_kernel = np.ones((3, 3)) / 9

        gradient = compute_gradient(self.testImage, avg_kernel)
        # Test the output.
        self.assertEqual(type(gradient), type(self.testImage))
        # Test
        self.assertEqual(gradient.shape, self.testImage.shape)

        print("\n\nSUCCESS: compute_gradient returns the correct output type.\n")

if __name__ == '__main__':
	unittest.main()

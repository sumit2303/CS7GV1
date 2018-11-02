# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:32:17 2018

@author: lenovo

Pattern detection using Histogram and back projection.
"""

import cv2
 
test_image = cv2.imread("map.jpg")
hsv_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
 
image_pattern = cv2.imread("green.jpg")
hsv_pattern = cv2.cvtColor(image_pattern, cv2.COLOR_BGR2HSV)
 
hue, saturation, value = cv2.split(hsv_pattern)
 
 
# Histogram of the pattern
pattern_hist = cv2.calcHist([hsv_pattern], [0, 1], None, [180, 256], [0, 180, 0, 256])

#Backprojection of test image from histogram of pattern
mask = cv2.calcBackProject([hsv_test], [0, 1], pattern_hist, [0, 180, 0, 256], 1)
 
# Filtering to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
 
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(test_image, mask)
 
cv2.imshow("Test image", test_image)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
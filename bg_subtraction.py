# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 23:33:08 2018

@author: lenovo

Real time object detection using background subtraction and contours
"""

import cv2
 
cap = cv2.VideoCapture("video.mp4")

 #Subtracting the static background
subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
 
while True:
    _, frame = cap.read()
 
    mask = subtractor.apply(frame)
    
    #contour creation on objects with a specific size
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
 
        if area > 500:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
 
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
 
    key = cv2.waitKey(30)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
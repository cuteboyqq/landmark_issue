# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:32:46 2023

@author: User
"""

import cv2
import numpy as np
import copy
import random as rng
import os
#======================================================================================================
#Before start coding~~~~~~~~~~~~~~~~~~~~~~
#========================Study image processing code===================================================
#====================================================================================================
'''Basic image processing (load/save/show image)''' 
#https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
'''Save roi image in python'''
#https://stackoverflow.com/questions/46879245/saving-contents-of-bounding-box-as-a-new-image-opencv-python
'''How to get contour by python cv2''' 
#https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
'''How to get mask'''
#https://stackoverflow.com/questions/32401806/get-mask-from-contour-with-opencv
'''Create BB of contour (Vary helpful code !)'''
#https://docs.opencv.org/4.x/da/d0c/tutorial_bounding_rects_circles.html

# 讀取圖檔
#==============================================================
#Step -1 : Filter the train images, Get the images with arrow landmark
#=============================================================
#========================
#Step 0 :loading image
#========================
raw_img = cv2.imread('./imgs/1326.jpg')
img = cv2.imread('./imgs/1326.jpg')
# 查看資料型態
print(type(img))
print(img.shape)
# 以灰階的方式讀取圖檔
#img = cv2.imread('1326.jpg')
  
# Setting All parameters
t_lower = 30  # Lower Threshold
t_upper = 150  # Upper threshold
#aperture_size = 7  # Aperture size
L2Gradient = True # Boolean
# Applying the Canny Edge filter
# with Custom Aperture Size
#edge = cv2.Canny(img, t_lower, t_upper, 
#                apertureSize=aperture_size)
#==================================
#Step 1 : Convert to Gray imag
#Step 2 : Get Binary image by thresohold
#==================================
blurred = cv2.GaussianBlur(img, (5, 5), 0) 
edge = cv2.Canny(blurred, t_lower, t_upper, L2gradient = L2Gradient ) #Get Canny image

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert BGR to Gray image
ret, thresh = cv2.threshold(imgray, 127, 255, 0) #imput Gray image, output Binary images (Mask)

kernel = np.ones((3, 3), np.uint8)
  
#==========================================
#Dilate or Erode the binary image
#=============================================
#thresh = cv2.dilate(thresh, kernel, iterations=2)
#thresh = cv2.erode(thresh, kernel, iterations=1)
#=========================================
#Step 3 : Using binary image to find contour
#======================================
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# show the images
#cv2.drawContours(img, contours, -1, (0,255,0), 3)

mask = np.zeros(img.shape, np.uint8)
for c in contours:
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(mask, c, -1, color , cv2.CHAIN_APPROX_SIMPLE)


contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
#===================================================
#Step 4 : Go through all contours
#===================================================
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly[i])
    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

drawing = np.ones((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
#========================================
#Create folders
#========================================
os.makedirs("./mask",exist_ok=True)
os.makedirs("./roi",exist_ok=True)
os.makedirs("./roi_2",exist_ok=True)
#==================================
#Step 5 :  Go through  all contours
#===================================
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    print("{}, {}, {}, {}".format(boundRect[i][0],boundRect[i][1],boundRect[i][2],boundRect[i][3]))
    x = boundRect[i][0]
    y = boundRect[i][1]
    w = boundRect[i][2]
    h = boundRect[i][3]
    print(edge.shape[0])
    print(edge.shape[1])
    #=============================================================
    # Filter out below two contours : 
    # 1.Small contour
    # 2.Contour is at the upper of the image
    # 3.Ratio is too small or too large
    #=====================================================================
    small_r = 0.13
    large_r = 7.0
    small_size = 600
    large_size = 12000
    if w*h > small_size and w*h < large_size and y> (edge.shape[0]*2.0/3.0) \
        and float(w/h)>small_r and float(w/h)<large_r:
        
        #====================================
        #Save Roi image and mask
        #====================================
        print(x*y)
        #================================
        # Save roi
        #=================================
        roi = raw_img[y:y+h, x:x+w]
        cv2.imwrite("roi/"+str(i)+".jpg", roi)
        #================================
        # Save mask
        #=================================
        binary = thresh[y:y+h, x:x+w]
        cv2.imwrite("mask/"+str(i)+".jpg", binary)
        #================================
        # Save roi without background
        #=================================
        roi_2 = raw_img[y:y+h, x:x+w]
        roi_2[binary==0] = 0 
        cv2.imwrite("roi_2/"+str(i)+".jpg", roi_2)
        
        #========================================================
        #Draw the final filtered contours with bounding boxes
        #=========================================================
        cv2.drawContours(img, contours_poly, i%255 , cv2.FILLED)
        cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
    (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)
     
    #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)


print("im gray {}".format(imgray.shape))

#========================================
#Visualize the processed images
#========================================
# 顯示圖片
cv2.imshow("th",thresh)
cv2.imshow('drawing',drawing)
cv2.imshow('My Image', img)
cv2.imshow('Mask', mask)
cv2.imshow('blur Image', blurred)
cv2.imshow('Gray Image', imgray)
cv2.imshow('Canny Image', edge)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
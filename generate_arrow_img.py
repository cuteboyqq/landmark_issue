# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 01:41:56 2023

@author: User
"""

import cv2
import os
import numpy as np
import random



img = cv2.imread("./raw_img/2076.jpg")
print(img.shape)

roi = cv2.imread("105.jpg")
roi_mask = cv2.imread("105_mask.jpg")

h_r = roi.shape[0]
w_r = roi.shape[1]


h = img.shape[0]
w = img.shape[1]
ratio_w = 1.4
ratio_h = 1.4
roi_l = np.ones((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)
roi_l_tmp = np.zeros((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)
img_roi = np.ones((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)


roi_s = np.ones((int(h_r/2.0),int(w_r/2.0), 3), dtype=np.uint8)
roi_s_tmp = np.ones((int(h_r/2.0),int(w_r/2.0), 3), dtype=np.uint8)
img_roi_s = np.ones((int(h_r/2.0),int(w_r/2.0), 3), dtype=np.uint8)

y = random.randint(int(h*float(4/5)),h)

Carhood = h*0.95

if y> h*5/6 and y<Carhood:
    print("case 1 ")
    roi_l = cv2.resize(roi,(int(w_r*ratio_w),int(h_r*ratio_h)),interpolation=cv2.INTER_NEAREST)
    roi_mask = cv2.resize(roi_mask,(int(w_r*ratio_w),int(h_r*ratio_h)),interpolation=cv2.INTER_NEAREST)
    x = random.randint(int(w*4/10),int(w*5/10))
    h_r = int(h_r*ratio_w)
    w_r = int(w_r*ratio_h)
    
    img_roi = img[y-h_r:y,x-w_r:x]
    roi_l_tmp[roi_mask>20] = roi_l[roi_mask>20]
    roi_l_tmp[roi_mask<=20] = img_roi[roi_mask<=20]
    
    #Wrong result, need to get rid of background
    img[y-h_r:y,x-w_r:x] = roi_l_tmp
    cv2.imshow("roi_l_tmp",roi_l_tmp)
elif y<= h*5/6 :
    print("case 2 ")
    x = random.randint(int(w*9/20),int(w*11/20))
    roi_mask = cv2.resize(roi_mask,(int(w_r/2.0),int(h_r/2.0)),interpolation=cv2.INTER_NEAREST)
    roi_s = cv2.resize(roi,(int(w_r/2.0),int(h_r/2.0)),interpolation=cv2.INTER_NEAREST)
    h_r = int(h_r/2.0)
    w_r = int(w_r/2.0)
    
    img_roi_s = img[y-h_r:y,x-w_r:x]
    roi_s_tmp[roi_mask>20] = roi_s[roi_mask>20]
    roi_s_tmp[roi_mask<=20] = img_roi_s[roi_mask<=20]
    cv2.imshow("img_roi_s",img_roi_s)
    img[y-h_r:y,x-w_r:x] = roi_s_tmp
    #Wrong result, need to get rid of background
    #img[y-h_r:y,x-w_r:x] = roi_s
else:
    print("at Carhood, pass!")






cv2.imshow("img",img)
cv2.imshow("roi",roi)
cv2.imshow("roi_mask",roi_mask)
# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

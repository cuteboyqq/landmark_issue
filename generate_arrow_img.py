# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 01:41:56 2023

@author: User
"""

import cv2
import os
import numpy as np
import random


def Generate_Landmark_Img(img_path=None,
                          roi_path=None,
                          roi_mask_path=None,
                          label_path=None):
    label = cv2.imread(label_path)
    label_gray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    _, label_mask = cv2.threshold(label_gray, 127, 255, 0)
    cv2.imshow("label_mask",label_mask)

    vanish_y = 0
    get_vanish = False
    for i in range(label_mask.shape[0]):
        if get_vanish == False:
            for j in range(label_mask.shape[1]):
                if label_mask[i][j]>0:
                    vanish_y = i
                    get_vanish = True
    
    print("get_vanish:{} ~{}".format(get_vanish,vanish_y))
    #input()

    carhood_y = label_mask.shape[0] -1 
    get_carhood_y = False
    for i in range(label_mask.shape[0]-1,0,-1):
        if not get_carhood_y:
            for j in range(label_mask.shape[1]):
                if label_mask[i][j]>0:
                    carhood_y = i
                    get_carhood_y = True

    carhood_y = carhood_y - int(label_mask.shape[0]/9.0)
    print("get_carhood_y:{} ~{}".format(get_carhood_y,carhood_y))
    #input()

    img = cv2.imread(img_path)
    print(img.shape)

    roi = cv2.imread(roi_path)
    roi_mask = cv2.imread(roi_mask_path)

    h_r = roi.shape[0]
    w_r = roi.shape[1]

    h = img.shape[0]
    w = img.shape[1]
    # ratio_w = 1.3
    # ratio_h = 1.3
    # roi_l = np.ones((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)
    # roi_l_tmp = np.zeros((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)
    # img_roi = np.ones((int(h_r*ratio_w),int(w_r*ratio_w), 3), dtype=np.uint8)

    # ratio_s = 2.0
    # roi_s = np.ones((int(h_r/ratio_s),int(w_r/ratio_s), 3), dtype=np.uint8)
    # roi_s_tmp = np.ones((int(h_r/ratio_s),int(w_r/ratio_s), 3), dtype=np.uint8)
    # img_roi_s = np.ones((int(h_r/ratio_s),int(w_r/ratio_s), 3), dtype=np.uint8)


    # Carhood = h*0.80
    y = random.randint(int(vanish_y),carhood_y)
    x = random.randint(int(w*0/10),int(w*10/10))
    while(label_mask[y][x]==0):
        x = random.randint(int(w*0/10),int(w*10/10))
        print("x in at background, re-random again~")
        #input()
    #============find the middle of drivable area=========================
    left_line_point_x = 0
    search_x = x
    while(search_x>0):
        if not label_mask[y][search_x]==0:
            search_x -=1
        elif label_mask[y][search_x]==0 :
            left_line_point_x = search_x
            break
    search_x = x
    right_line_point_x = 0
    while(search_x<label_mask.shape[1]):
        if not label_mask[y][search_x]==0:
            search_x +=1
        elif label_mask[y][search_x]==0 :
            right_line_point_x = search_x
            break

    print("left_line_point_x:{}".format(left_line_point_x))
    print("right_line_point_x:{}".format(right_line_point_x))
    final_x = int((left_line_point_x + right_line_point_x )/2.0)
    print("final_x:{}".format(final_x))
    print("final_y:{}".format(y))
    road_width = abs(right_line_point_x - left_line_point_x)
    print("road_width = {}".format(road_width))
  
    roi_w, roi_h = roi.shape[1], roi.shape[0]

    final_roi_w = road_width * 0.7
    resize_ratio = float(final_roi_w/roi_w)
    final_roi_h = int(roi_h * resize_ratio)
    #roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)

    roi_l = np.ones((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_l_tmp = np.zeros((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)
    img_roi = np.ones((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)

   

    
    if y> (vanish_y)+1 and y<carhood_y-1:
        print("case 1 ")
        roi_l = cv2.resize(roi,(int(w_r*resize_ratio),int(h_r*resize_ratio)),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(int(w_r*resize_ratio),int(h_r*resize_ratio)),interpolation=cv2.INTER_NEAREST)
        x = final_x
        h_r = int(h_r*resize_ratio)
        print("h_r = {}".format(h_r))
        h_add = 0
        if h_r%2!=0:
           h_add = 1

        w_r = int(w_r*resize_ratio)
        print("w_r = {}".format(w_r))
        w_add = 0
        if w_r%2!=0:
            w_add = 1
        #input()
        img_roi = img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
        print("roi_l_tmp {}".format(roi_l_tmp.shape))
        print("img_roi {}".format(img_roi.shape))
        roi_l_tmp[roi_mask>20] = roi_l[roi_mask>20]
        roi_l_tmp[roi_mask<=20] = img_roi[roi_mask<=20]
        
        #Wrong result, need to get rid of background
        img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_l_tmp
        cv2.imshow("roi_l_tmp",roi_l_tmp)
    
    else:
        print("at Carhood, pass!")

    os.makedirs("./fake_image",exist_ok=True)
    cv2.imwrite("./fake_image/fake_img.jpg",img)

    cv2.imshow("img",img)
    cv2.imshow("roi",roi)
    cv2.imshow("roi_mask",roi_mask)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":

    img_path = "./imgs/b2de6f59-9f74dea1.jpg"
    roi_path = "./roi/113.jpg"
    roi_mask_path = "./mask/113.jpg"
    label_path = "./labels/b2de6f59-9f74dea1.png"
    Generate_Landmark_Img(img_path,
                          roi_path,
                          roi_mask_path,
                          label_path)

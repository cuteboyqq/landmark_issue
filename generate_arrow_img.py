# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 01:41:56 2023

@author: User
"""

import cv2
import os
import numpy as np
import random
import glob
import random

def Analysis_path(img_path):
    img = img_path.split("/")[-1]
    img_name = img.split(".")[0]
    return img, img_name

def Generate_Landmark_Img(img_path=None,
                          roi_path=None,
                          roi_mask_path=None,
                          label_path=None,
                          line_label_path=None,
                          save_landmark_img=False,
                          show_img=False):
    label = cv2.imread(label_path)
    line_label = cv2.imread(line_label_path)
    label_gray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    _, label_mask = cv2.threshold(label_gray, 127, 255, 0)
    #cv2.imshow("label_mask",label_mask)
    #cv2.imshow("label",label)
    vanish_y = 0
    get_vanish = False
    # print(label_mask.shape[0])
    # print(label_mask.shape[1])
    # input()
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

    carhood_y = carhood_y - int(label_mask.shape[0]/7.0)
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
    if vanish_y < carhood_y:
        y = random.randint(int(vanish_y),carhood_y)
    else:
        y = random.randint(int(img.shape[0]/2.0),int(img.shape[0]*5.0/6.0))

    x = random.randint(int(w*2/10),int(w*8/10))
    co = 1
    while(label_mask[y][x]==0):
        x = random.randint(int(w*2/10),int(w*8/10))
        print("x in at background, re-random again~")
        co+=1
        if co==100:
            break
        #input()
    #============find the middle of drivable area=========================
    left_line_point_x = 0
    search_x = x
    #(r,g,b) = label[y][x]
    print(label[y][x])
    while(search_x>0):
        if label[y][search_x][0]== label[y][x][0]:
            search_x -=1
        elif not label[y][search_x][0]== label[y][x][0] :
            left_line_point_x = search_x
            break

    print("left_line_point_x:{}".format(left_line_point_x))
    search_x = x
    right_line_point_x = 0
    while(search_x<label_mask.shape[1]):
        if label[y][search_x][0]==label[y][x][0]:
            search_x +=1
        elif not label[y][search_x][0]==label[y][x][0] :
            right_line_point_x = search_x
            break
        elif label[y][search_x][0]==label[y][x][0] and search_x==label_mask.shape[1]*5.0/6.0:
            right_line_point_x = search_x

    print("right_line_point_x:{}".format(right_line_point_x))


    final_x = int((left_line_point_x + right_line_point_x )/2.0)
    print("final_x:{}".format(final_x))
    print("final_y:{}".format(y))
    road_width = abs(right_line_point_x - left_line_point_x)
    print("road_width = {}".format(road_width))
    

    roi_w, roi_h = roi.shape[1], roi.shape[0]

    final_roi_w = road_width * 0.40
    resize_ratio = float(final_roi_w/roi_w)
    final_roi_h = int(roi_h * resize_ratio)
    if road_width == 0 or float(final_roi_w/roi_w)==0 or resize_ratio==0:
        return
    
    #roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)

    roi_l = np.ones((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_l_tmp = np.zeros((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)
    img_roi = np.ones((int(h_r*resize_ratio),int(w_r*resize_ratio), 3), dtype=np.uint8)

   

    
    if y> (vanish_y)+ abs(carhood_y-vanish_y)/10.0 and y<carhood_y-1:
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
        #filter landmark location at line area 2023-08-08
        is_line_area = False
        use_line_label = True
        if use_line_label:
            for i in range(y-int(h_r/2.0),y+int(h_r/2.0)):
                for j in range(x-int(w_r/2.0),x+int(w_r/2.0)):
                    if line_label[i][j][0]<255 and line_label[i][j][1]<255 and line_label[i][j][2]<255:
                        is_line_area=True

        if not is_line_area:  
            img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_l_tmp
        else:
            print("ROI is at line label area, skip~~")
            #input()
        #cv2.imshow("roi_l_tmp",roi_l_tmp)
    
    else:
        print("at Carhood, pass!")

    
    if save_landmark_img:
        image, img_name = Analysis_path(img_path)
        landmark_img = image
        os.makedirs("./fake_landmark_image",exist_ok=True)
        cv2.imwrite(os.path.join("./fake_landmark_image/",landmark_img),img)
    if show_img:
        cv2.imshow("img",img)
        cv2.imshow("roi",roi)
        cv2.imshow("roi_mask",roi_mask)
        #按下任意鍵則關閉所有視窗
        cv2.waitKey(1)
        cv2.destroyAllWindows()


def Generate_landmark_Imgs(img_dir=None,
                           label_dir=None,
                           line_label_dir=None,
                           roi_dir=None,
                           roi_mask_dir=None,
                           save_landmark_img=True,
                           generate_number=None,
                           show_img=False):
    img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    label_path_list = glob.glob(os.path.join(label_dir,"*.png"))
    roi_path_list = glob.glob(os.path.join(roi_dir,"*.jpg"))
    mask_path_list = glob.glob(os.path.join(roi_dir,"*.jpg"))

    

    print(img_path_list)
    c = 0
    for img_path in img_path_list:
        #random choose landmark mask
        selected_roi_path = roi_path_list[random.randint(0,len(roi_path_list)-1)]
        r, r_name = Analysis_path(selected_roi_path)
        selected_mask_path = os.path.join(roi_mask_dir,r)

        print(img_path)
        img, img_name = Analysis_path(img_path)
        label = img_name + ".png"
        label_path = os.path.join(label_dir,label)
        line_label_path = os.path.join(line_label_dir,label)
        print(label_path)
        # image = cv2.imread(img_path)
        # la = cv2.imread(label_path)
        # cv2.imshow("img",image)
        # cv2.imshow("label",la)
        # cv2.waitKey(0)
        c+=1
        print(c)
        if c==int(generate_number):
            break
        Generate_Landmark_Img(img_path=img_path,
                          roi_path=selected_roi_path,
                          roi_mask_path=selected_mask_path,
                          label_path=label_path,
                          line_label_path=line_label_path,
                          save_landmark_img=save_landmark_img,
                          show_img=show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":

    img_path = "./imgs/b4dd1c23-355940ff.jpg"
    roi_path = "./roi/113.jpg"
    roi_mask_path = "./mask/113.jpg"
    label_path = "./labels/b4dd1c23-355940ff.png"
    line_label_path = "./line_label/b4dd1c23-355940ff.png"
    # Generate_Landmark_Img(img_path,
    #                       roi_path,
    #                       roi_mask_path,
    #                       line_label_path,
    #                       label_path)

    img_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/images/100k/train"
    label_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/drivable/colormaps/train"
    line_label_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/lane/masks/train"
    roi_dir = "/home/jnr_loganvo/Alister/GitHub_Code/landmark_issue/roi"
    roi_mask_dir = "/home/jnr_loganvo/Alister/GitHub_Code/landmark_issue/mask"
    save_landmark_img = True
    generate_number = 10000
    show_img = False
    Generate_landmark_Imgs(img_dir,
                           label_dir,
                           line_label_dir,
                           roi_dir,
                           roi_mask_dir,
                           save_landmark_img,
                           generate_number,
                           show_img)
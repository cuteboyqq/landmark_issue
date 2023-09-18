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
import logging
import numpy as np
logger = logging.getLogger('my-logger')
logger.propagate = False

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def Analysis_path(img_path):
    img = img_path.split("/")[-1]
    img_name = img.split(".")[0]
    return img, img_name

def Generate_Landmark_Img(img_path=None,
                          roi_path=None,
                          roi_mask_path=None,
                          label_path=None,
                          label_train_path=None,
                          line_label_path=None,
                          save_landmark_img=False,
                          save_colormap=False,
                          save_mask=False,
                          save_txt=False,
                          show_img=False,
                          use_mask=True,
                          use_opencv_ratio=0.25):
    #USE_OPENCV = True #Force using opencv~~~
    IS_FAILED = False
    label_colormap = cv2.imread(label_path) #drivable area label colomap
    #print("[Generate_Landmark_Img]label_train_path:{}".format(label_train_path))
    
    label_train = cv2.imread(label_train_path) #drivable area label mask for train
    # for i in range(label_train.shape[0]):
    #     for j in range(label_train.shape[1]):
    #         print(label_train[i][j])
    #label_train = cv2.cvtColor(label_train,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("label_train",label_train)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    print("label_train shape :{}".format(label_train.shape))
    #input()
    line_label = cv2.imread(line_label_path)
    label_gray = cv2.cvtColor(label_colormap,cv2.COLOR_BGR2GRAY)
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

    # Carhood = h*0.80
    if vanish_y < carhood_y:
        y = random.randint(int(vanish_y),carhood_y-1) # Get the final  ROI y
    else:
        y = random.randint(int(img.shape[0]/2.0),int(img.shape[0]*5.0/6.0)) # Get the final  ROI y

    x = random.randint(int(w*3/10),int(w*7/10))
    co = 1
    while(label_mask[y][x]==0):
        x = random.randint(int(w*3/10),int(w*7/10)) # Get the final  ROI x
        print("x in at background, re-random again~")
        co+=1
        if co==150:
            IS_FAILED = True
            return IS_FAILED
        #input()
    #============find the middle of drivable area=========================
    left_line_point_x = 0
    search_x = x
    #(r,g,b) = label[y][x]
    print(label_colormap[y][x])
    while(search_x>0):
        if label_colormap[y][search_x][0]== label_colormap[y][x][0]:
            search_x -=1
        elif not label_colormap[y][search_x][0]== label_colormap[y][x][0] :
            left_line_point_x = search_x
            break

    print("left_line_point_x:{}".format(left_line_point_x))
    search_x = x
    right_line_point_x = left_line_point_x + 2
    while(search_x<label_mask.shape[1]):
        if label_colormap[y][search_x][0]==label_colormap[y][x][0]:
            search_x +=1
        elif not label_colormap[y][search_x][0]==label_colormap[y][x][0] :
            right_line_point_x = search_x
            break
        elif label_colormap[y][search_x][0]==label_colormap[y][x][0] and search_x==label_mask.shape[1]*5.0/6.0:
            right_line_point_x = search_x

    print("right_line_point_x:{}".format(right_line_point_x))


    final_x = int((left_line_point_x + right_line_point_x )/2.0)
    print("final_x:{}".format(final_x))
    print("final_y:{}".format(y))
    road_width = abs(right_line_point_x - left_line_point_x)
    print("road_width = {}".format(road_width))
    

    roi_w, roi_h = roi.shape[1], roi.shape[0]
    #Set landmark width (initial setting)
    final_roi_w = road_width * 0.50 #Get the final ROI width

    print("final_roi_w:{}".format(final_roi_w))
    resize_ratio = float(final_roi_w/roi_w)
    #resize_ratio_h = float(road_width * 0.60/roi_w)
    print("initial resize_ratio:{}".format(resize_ratio))
    if road_width <= img.shape[1]*3/5:
        resize_ratio = float(final_roi_w/roi_w)
        print("resize_ratio case 1 ")
    else:
        final_roi_w = road_width * 0.30 #Get the final ROI width
        resize_ratio = float(final_roi_w/roi_w) #if no line, lane width is too large
        print("resize_ratio case 2 ")

    print("resize_ratio:{}".format(resize_ratio))
    resize_ratio_h = resize_ratio
    final_roi_h = int(roi_h * resize_ratio) #Get the final ROI height

    
        
    if road_width <  10 or float(final_roi_w/roi_w)==0 or resize_ratio==0:
        IS_FAILED=True
        return IS_FAILED
    
    #roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
    roi_l = np.ones((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_l_tmp = np.zeros((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_diff = np.zeros((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_tmp_v1 = np.zeros((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_tmp_v2 = np.zeros((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    roi_l_tmp_train = np.zeros((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    img_roi = np.ones((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)
    label_roi = np.ones((int(h_r*resize_ratio_h),int(w_r*resize_ratio), 3), dtype=np.uint8)

    (opencv_ratio_count) = int(use_opencv_ratio * 100) #0.25*100=25
    USE_OPENCV = True
    (rand_num) = random.randint(1,100) 
    USE_OPENCV = True if (rand_num)<=(opencv_ratio_count) else False #(opencv_ratio_count)% using opencv
    
    if use_mask==False:
        USE_OPENCV=True

    if USE_OPENCV:
        if y> (vanish_y)+ abs(carhood_y-vanish_y)/10.0 and y<carhood_y-1:
            try:
                roi_l = cv2.resize(roi,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
                roi_mask_l = cv2.resize(roi_mask,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
                roi_mask_l = cv2.cvtColor(roi_mask_l, cv2.COLOR_BGR2GRAY) #Convert BGR to Gray image
                ret, roi_mask_l = cv2.threshold(roi_mask_l, 150, 255, 0) #imput Gray image, output Binary images (Mask)
                contours, hierarchy = cv2.findContours(roi_mask_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                print("roi_l = cv2.resize Error !")
                IS_FAILED=True
                return IS_FAILED
            #contours_poly = [None]*len(contours)
            # for i, c in enumerate(contours):
            #     contours_poly[i] = cv2.approxPolyDP(c,3, True)#3
            #===================filter landmark location at line area 2023-08-08=====================================================================================
            is_line_area = False
            use_line_label = True
            x = final_x
            y_uper = y+int(h_r/2.0) if y+int(h_r/2.0) < img.shape[0] else img.shape[0] - 1
            y_lower = y-int(h_r/2.0) if y-int(h_r/2.0) > 0 else 0
           
            x_uper = x+int(w_r/2.0) if x+int(w_r/2.0) < img.shape[1] else img.shape[1] - 1
            x_lower = x-int(w_r/2.0) if x-int(w_r/2.0) >= 0 else 0
           
            out_of_range = False
            if use_line_label:
                for i in range(y_lower,y_uper):
                    for j in range(x_lower,x_uper):
                        if line_label[i][j][0]<255 or line_label[i][j][1]<255 or line_label[i][j][2]<255:
                            is_line_area=True
                            #print("USE_OPENCV is_line_area=True")
                            #input()
                            IS_FAILED = True
                            return IS_FAILED
                        if label_mask[i][j] == 0:
                            out_of_range = True
            #============================================================================================================================
            h_r = int(h_r*resize_ratio_h)
            print("h_r = {}".format(h_r))
            h_add = 0
            if h_r%2!=0:h_add = 1
            w_r = int(w_r*resize_ratio)
            print("w_r = {}".format(w_r))
            w_add = 0
            if w_r%2!=0:w_add = 1

            try:
                img_roi = img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
                label_roi = label_train[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
                colormap_roi = label_colormap[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
            except:
                IS_FAILED=True
                print("fail at label_roi = label_train[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]")
                #input()
                return IS_FAILED
            
            use_mask_method=True #use new method if mask is not beautiful to separate forground and background
            try:
                roi_l_tmp[roi_mask_l>20] = roi_l[roi_mask_l>20] #put the landmark into image roi
                roi_l_tmp[roi_mask_l<=20] = img_roi[roi_mask_l<=20] #keep the background in image roi


                if use_mask_method:
                    # Use mask to get re-label the label.png, if you have clear roi_mask
                    roi_l_tmp_train[roi_mask_l>20] = 3 #assign new label landmark=3
                    roi_l_tmp_train[roi_mask_l<=20] = label_roi[roi_mask_l<=20] #keep the original label of background
                    #color map
                    roi_tmp_v1[roi_mask_l>20] =  (255,127,0)#assign new label landmark=3
                    roi_tmp_v1[roi_mask_l<=20] = colormap_roi[roi_mask_l<=20] #keep the original label of background
                else: #Non-mask method to generatee new label.png #Result is not good
                    roi_diff = roi_l_tmp - img_roi #find the difference of ori-imge and new-image
                    roi_tmp_v2[roi_diff>10] = 150 #Test using 150 for view image
                    roi_tmp_v2[roi_diff<=10] = label_roi[roi_diff<=10]
            except:
                IS_FAILED=True
                return IS_FAILED
            
            #Use try-except to ignore errors : 
            # https://stackoverflow.com/questions/38707513/ignoring-an-error-message-to-continue-with-the-loop-in-python
            if not is_line_area:
                try:
                     # output1 = img
                    mask = 255 * np.ones(roi_l.shape, roi_l.dtype)
                    center = (final_x,y)                
                    #for i in range(100):
                    output1 = cv2.seamlessClone(roi_l, img, mask, center, cv2.MIXED_CLONE)                                           
                    label_train[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_l_tmp_train \
                        if use_mask_method else roi_tmp_v2 #new label for landmark
                    label_colormap[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_tmp_v1 
                except:
                    output1 = img
                    print("output1 = cv2.seamlessClone(roi_l, img, mask, center, cv2.MIXED_CLONE) Failed ")
                    #input()
                    IS_FAILED = True
                    return IS_FAILED
            else:
                output1 = img
                IS_FAILED = True
                return IS_FAILED
            if show_img:
                cv2.imshow("output1",output1) #img
            if save_landmark_img and not IS_FAILED:

                #Now we have information of ROI coordinate : x,y,w,h , so we can save it into filename.txt of yolo format
                if save_txt:
                    os.makedirs("./fake_landmark_image_test/label",exist_ok=True)
                    image, img_name = Analysis_path(img_path)
                    landmark_label = 10
                    x = float(int((final_x/img.shape[1])*1000000)/1000000)
                    y = float(int((y/img.shape[0])*1000000)/1000000)
                    w = float(int((final_roi_w/img.shape[1])*1000000)/1000000)
                    h = float(int((final_roi_h/img.shape[0])*1000000)/1000000)
                    use_bdd100k_dataset = True
                    if use_bdd100k_dataset:
                        try:
                            f = open("/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train/"+img_name+".txt",'r')
                        
                            f2 = open("./fake_landmark_image_test/label/"+img_name+".txt",'w')
                            f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                            f2.write("\n")
                            for line in f.readlines():
                                f2.write(line)

                            f2.close()
                            f.close()
                        except:
                            print("Pass saving labels.txt,No bdd100k label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED
                    else:
                        f2 = open("./runs/predict/label/"+img_name+".txt",'w')
                        f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                        f2.close()

                image, img_name = Analysis_path(img_path)
                landmark_img = image
                label = img_name + ".png"
                image_dir = os.path.join("./fake_landmark_image_test","images")
                
                os.makedirs(image_dir,exist_ok=True)
                
                cv2.imwrite("./fake_landmark_image_test/images/"+landmark_img,output1)
                if save_mask:
                    label_dir = os.path.join("./fake_landmark_image_test","masks")
                    os.makedirs(label_dir,exist_ok=True)
                    cv2.imwrite("./fake_landmark_image_test/masks/"+label,label_train)
                if save_colormap:
                    colormap_dir = os.path.join("./fake_landmark_image_test","colormaps")
                    os.makedirs(colormap_dir,exist_ok=True)
                    cv2.imwrite("./fake_landmark_image_test/colormaps/"+label,label_colormap)
                # landmark_img = image
                # os.makedirs("./fake_landmark_image_test",exist_ok=True)
                # cv2.imwrite(os.path.join("./fake_landmark_image_test/",landmark_img),output1)
        else:
            print("at Carhood, pass!")
    else: #Use Mask method
        try:
            roi_mask_l = cv2.resize(roi_mask,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
            roi_mask_l = cv2.cvtColor(roi_mask_l, cv2.COLOR_BGR2GRAY) #Convert BGR to Gray image
            ret, roi_mask_l = cv2.threshold(roi_mask_l, 150, 255, 0) #imput Gray image, output Binary images (Mask)
        except:
            print("roi_mask_l = cv2.resize Error~~~")
            IS_FAILED=True
            return IS_FAILED
        
        if y> (vanish_y)+ abs(carhood_y-vanish_y)/10.0 and y<carhood_y-1:
            print("case 1 ")
            try:
                roi_l = cv2.resize(roi,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
                roi_mask = cv2.resize(roi_mask,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
            except:
                IS_FAILED=True
                return IS_FAILED
            x = final_x
            h_r = int(h_r*resize_ratio_h)
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
            try:
                img_roi = img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
                label_roi = label_train[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
                colormap_roi = label_colormap[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add]
            except:
                IS_FAILED=True
                return IS_FAILED
            print("roi_l_tmp {}".format(roi_l_tmp.shape))
            print("img_roi {}".format(img_roi.shape))
            try:
                roi_l_tmp[roi_mask>20] = roi_l[roi_mask>20] # put landmark into image
                roi_l_tmp[roi_mask<=20] = img_roi[roi_mask<=20] #keep the background pixel in image
                # names_drive:
                # 0: direct area
                # 1: alternative area
                # 2: background
                # 3: landmark
                roi_l_tmp_train[roi_mask>20] = 3 #0:
                roi_l_tmp_train[roi_mask<=20] = label_roi[roi_mask<=20]
                #color map
                roi_tmp_v1[roi_mask_l>20] = (255,127,0)#assign new label landmark=3
                roi_tmp_v1[roi_mask_l<=20] = colormap_roi[roi_mask_l<=20] #keep the original label of background
            except:
                print("roi_l_tmp_train error~~")
                #input()
                IS_FAILED=True
                return
        
            #Wrong result, need to get rid of background
            #filter landmark location at line area 2023-08-08
            is_line_area = False
            use_line_label = True
            x = final_x
            y_uper = y+int(h_r/2.0) if y+int(h_r/2.0) < img.shape[0] else img.shape[0] - 1
            y_lower = y-int(h_r/2.0) if y-int(h_r/2.0) > 0 else 0
           
            x_uper = x+int(w_r/2.0) if x+int(w_r/2.0) < img.shape[1] else img.shape[1] - 1
            x_lower = x-int(w_r/2.0) if x-int(w_r/2.0) >= 0 else 0
         
            out_of_range = False
            if use_line_label:
                for i in range(y_lower,y_uper):
                    for j in range(x_lower,x_uper):
                        if line_label[i][j][0]<255 or line_label[i][j][1]<255 or line_label[i][j][2]<255:
                            is_line_area=True
                            IS_FAILED = True
                            return IS_FAILED
                        if label_mask[i][j] == 0:
                            out_of_range = True
    
            if not is_line_area:
                try:
                    img[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_l_tmp
                    label_train[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_l_tmp_train #new label for landmark
                    label_colormap[y-int(h_r/2.0):y+int(h_r/2.0)+h_add,x-int(w_r/2.0):x+int(w_r/2.0)+w_add] = roi_tmp_v1 #new colormap for landmark
                except:
                    IS_FAILED=True
                    #print("label_train IS_FAILED")
                    #input()
                    return IS_FAILED
            else:
                print("ROI is at line label area, skip~~")
                IS_FAILED=True
            #cv2.imshow("roi_l_tmp",roi_l_tmp)
            if show_img:
                cv2.imshow("img",img) #img

            if save_landmark_img and not IS_FAILED:

                #Now we have information of ROI coordinate : x,y,w,h , so we can save it into filename.txt of yolo format
                if save_txt: #save yolo label.txt
                    os.makedirs("./fake_landmark_image_test/label",exist_ok=True)
                    image, img_name = Analysis_path(img_path)
                    landmark_label = 10
                    x = float(int((final_x/img.shape[1])*1000000)/1000000)
                    y = float(int((y/img.shape[0])*1000000)/1000000)
                    w = float(int((final_roi_w/img.shape[1])*1000000)/1000000)
                    h = float(int((final_roi_h/img.shape[0])*1000000)/1000000)
                    use_bdd100k_dataset = True
                    if use_bdd100k_dataset:
                        try:
                            f = open("/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train/"+img_name+".txt",'r')
                        
                            f2 = open("./fake_landmark_image_test/label/"+img_name+".txt",'w')
                            f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                            f2.write("\n")
                            for line in f.readlines():
                                f2.write(line)

                            f2.close()
                            f.close()
                        except:
                            print("Pass saving labels.txt,No bdd100k label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED
                    else:
                        f2 = open("./runs/predict/label/"+img_name+".txt",'w')
                        f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                        f2.close()


                image, img_name = Analysis_path(img_path)
                landmark_img = image
                label = img_name + ".png"
                image_dir = os.path.join("./fake_landmark_image_test","images")
                label_dir = os.path.join("./fake_landmark_image_test","masks")
                os.makedirs(image_dir,exist_ok=True)
                os.makedirs(label_dir,exist_ok=True)
                cv2.imwrite("./fake_landmark_image_test/images/"+landmark_img,img)
                cv2.imwrite("./fake_landmark_image_test/masks/"+label,label_train)
                if save_colormap:
                    colormap_dir = os.path.join("./fake_landmark_image_test","colormaps")
                    os.makedirs(colormap_dir,exist_ok=True)
                    cv2.imwrite("./fake_landmark_image_test/colormaps/"+label,label_colormap)
               
        else:
            print("at Carhood, pass!")

    
    
    if show_img:
        #cv2.imshow("img",img)
        cv2.imshow("roi",roi)
        cv2.imshow("roi_mask",roi_mask)
        #按下任意鍵則關閉所有視窗
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    return IS_FAILED


def Generate_landmark_Imgs(img_dir=None,
                           label_dir=None,
                           label_dir_train=None,
                           line_label_dir=None,
                           roi_dir=None,
                           roi_mask_dir=None,
                           save_landmark_img=True,
                           save_colormap=True,
                           save_mask=True,
                           save_txt=True,
                           generate_number=None,
                           show_img=False,
                           use_mask=True,
                           use_opencv_ratio=0.25):
    img_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    label_path_list = glob.glob(os.path.join(label_dir,"*.png"))
    label_train_path_list = glob.glob(os.path.join(label_dir_train,"*.png"))
    roi_path_list = glob.glob(os.path.join(roi_dir,"*.jpg"))
    mask_path_list = glob.glob(os.path.join(roi_dir,"*.jpg"))

    

    print(img_path_list)
    c = 0
    for img_path in img_path_list:
        
        print(img_path)
        img, img_name = Analysis_path(img_path)
        label = img_name + ".png"
        label_path = os.path.join(label_dir,label)
        label_train_path = os.path.join(label_dir_train,label)
        #print("label_train_path:{}".format(label_train_path))
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
        IS_FAILED = True
        count = 1
        while(IS_FAILED and count!=10):
            #random choose landmark mask
            selected_roi_path = roi_path_list[random.randint(0,len(roi_path_list)-1)]
            r, r_name = Analysis_path(selected_roi_path)
            selected_mask_path = os.path.join(roi_mask_dir,r)

            IS_FAILED = Generate_Landmark_Img(img_path=img_path,
                            roi_path=selected_roi_path,
                            roi_mask_path=selected_mask_path,
                            label_path=label_path,
                            label_train_path=label_train_path,
                            line_label_path=line_label_path,
                            save_landmark_img=save_landmark_img,
                            save_colormap=save_colormap,
                            save_mask=save_mask,
                            save_txt=save_txt,
                            show_img=show_img,
                            use_mask=use_mask,
                            use_opencv_ratio=0.25)
                            
            count+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-drilabeldir','--dri-labeldir',help='drivable label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    parser.add_argument('-drilabeldirtrain','--dri-labeldirtrain',help='drivable label dir fo train',default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/masks/train")
    parser.add_argument('-linelabeldir','--line-labeldir',help='line label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/lane/masks/train")
    parser.add_argument('-roidir','--roi-dir',help='roi dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/roi")
    parser.add_argument('-roimaskdir','--roi-maskdir',help='roi mask dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/mask")
    parser.add_argument('-saveimg','--save-img',action='store_true',help='save landmark fake images')
    parser.add_argument('-savecolormap','--save-colormap',action='store_true',help='save generate semantic segment colormaps')
    parser.add_argument('-savemask','--save-mask',action='store_true',help='save generate semantic segment train masks')
    parser.add_argument('-savetxt','--save-txt',action='store_true',help='save landmark fake label.txt in yolo format cxywh')
    parser.add_argument('-numimg','--num-img',type=int,default=30000,help='number of generate fake landmark images')
    parser.add_argument('-useopencvratio','--use_opencvratio',type=float,default=0.50,help='ratio of using opencv method to generate landmark images')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='use mask method to generate landmark or not')
    parser.add_argument('-show','--show',action='store_true',help='show images result')
   
    return parser.parse_args()    


if __name__=="__main__":

    img_path = "./datasets/imgs/b4dd1c23-355940ff.jpg"
    roi_path = "./roi/45.jpg"
    roi_mask_path = "./mask/45.jpg"
    label_path = "./datasets/labels/b4dd1c23-355940ff.png"
    line_label_path = "./datasets/line_label/b4dd1c23-355940ff.png"

    INFER_ONE_IMG=False
    if INFER_ONE_IMG:
        Generate_Landmark_Img(img_path=img_path,
                                roi_path=roi_path,
                                roi_mask_path=roi_mask_path,
                                label_path=label_path,
                                line_label_path=line_label_path,
                                save_landmark_img=True,
                                show_img=True)

    args = get_args()
    img_dir = args.img_dir
    label_dir = args.dri_labeldir
    label_dir_train = args.dri_labeldirtrain
    line_label_dir = args.line_labeldir
    roi_dir = args.roi_dir
    roi_mask_dir = args.roi_maskdir
    save_landmark_img = True
    save_txt = True #args.save_txt
    generate_number = args.num_img
    use_mask = args.use_mask
    show_img = False
    save_colormap = True #args.save_colormap
    save_mask = True #args.save_mask
    use_opencv_ratio = args.use_opencvratio

    # img_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/images/100k/train"
    # label_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/drivable/colormaps/train"
    # line_label_dir = "/home/jnr_loganvo/Alister/datasets/YOLO_ADAS/bdd100k_data/labels/lane/masks/train"
    # roi_dir = "/home/jnr_loganvo/Alister/GitHub_Code/landmark_issue/roi"
    # roi_mask_dir = "/home/jnr_loganvo/Alister/GitHub_Code/landmark_issue/mask"
    # save_landmark_img = True
    # generate_number = 10000
    # show_img = False
    USE_FOLDER_DATASET=True
    if USE_FOLDER_DATASET:
        Generate_landmark_Imgs(img_dir,
                            label_dir,
                            label_dir_train,
                            line_label_dir,
                            roi_dir,
                            roi_mask_dir,
                            save_landmark_img,
                            save_colormap,
                            save_mask,
                            save_txt,
                            generate_number,
                            show_img,
                            use_mask,
                            use_opencv_ratio)
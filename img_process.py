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
import shutil

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import glob
from PIL import Image
import torchvision.transforms as transforms
from util.load_model import load_model
import shutil
from matplotlib import cm
import glob

if os.path.exists("./mask_dirty"):
    shutil.rmtree('./mask_dirty')
if os.path.exists("./roi_dirty"):
    shutil.rmtree('./roi_dirty')
if os.path.exists("./roi_2_dirty"):   
    shutil.rmtree('./roi_2_dirty')
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



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-datatest','--data-test',help='custom test data)',\
        default=r'/home/ali/Projects/GitHub_Code/ali/landmark_issue')
    parser.add_argument('-imgdir','--img-dir',help='image directory that you want to processing)',\
        default=r'/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/landmark_img')
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=64)
    parser.add_argument('-nc','--nc',type=int,help='num of channels',default=3)
    parser.add_argument('-model','--model',help='resnet,VGG16,repvgg,res2net',default='resnet')
    parser.add_argument('-mpath','--model-path',help='pretrained model path',\
        default=r'/home/ali/Projects/GitHub_Code/ali/landmark_issue/runs/train/resnet_best.pt')
    parser.add_argument('-viewimg','--view-img',type=bool,help='view process result images',default=False)
    parser.add_argument('-saveroi','--save-roi',type=bool,help='save landmark roi images',default=True)
    parser.add_argument('-savemask','--save-mask',type=bool,help='save landmark mask images',default=True)
    return parser.parse_args()

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opts = get_args()
# 讀取圖檔
#==============================================================
#Step -1 : Filter the train images, Get the images with arrow landmark
#=============================================================
def Get_Arrow_ROI(img_path = './datasets/landmark_img/Screenshot from 2023-08-10 12-50-26.png',
                  view_img = True,
                  save_roi = True,
                  save_mask = True,
                  count=None):
    #========================
    #Step 0 :loading image
    #========================
    raw_img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    # 查看資料型態
    print(type(img))
    #print(img.shape)

    img_h = img.shape[0]
    img_w = img.shape[1]
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

    ret, thresh = cv2.threshold(imgray, 180, 255, 0) #imput Gray image, output Binary images (Mask)

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
    # for c in contours:
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        
    #     area = cv2.contourArea(c)
    #     print("area: {}".format(area))
    #     if area > 400:
    #         cv2.drawContours(mask, c, -1, color , cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    #===================================================
    #Step 4 : Go through all contours
    #===================================================
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c,3, True)#3
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    drawing = np.ones((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
   
    os.makedirs("./mask_dirty",exist_ok=True)
    os.makedirs("./roi_dirty",exist_ok=True)
    #os.makedirs("./roi_2",exist_ok=True)

    #============load pre-trained model================================
    #model = load_model(opts,opts.nc) #For example : model = ResNet(ResBlock,nc=nc)
    #print('model :{}'.format(opts.model))

    #model.load_state_dict(torch.load(opts.model_path)) #load pre-trained model
    print(opts.model_path)
    model = torch.load(opts.model_path)
    #print(model)
    if torch.cuda.is_available():
        model.cuda()

    pre_process = transforms.Compose([
                    transforms.Resize(opts.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(opts.img_size),
                    transforms.ToTensor()
                    #normalize
                    ])

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
        small_r = 0.01
        large_r = 100.0
        small_size = 200#img_h*img_w/100
        large_size = 110000#img_h*img_w/1
        print("w*h:{}".format(w*h))
        if w*h > small_size and w*h < large_size and y> (edge.shape[0]*1.0/5.0) \
            and float(w/h)>small_r and float(w/h)<large_r:
            
            print(x*y)
           
            roi = raw_img[y:y+h, x:x+w]
            print("{} roi".format(count))
            index = i
            if save_roi:
                cv2.imwrite("roi_dirty/"+str(count)+".jpg", roi)  # Save roi
            
            binary = thresh[y:y+h, x:x+w]
            if save_mask:
                cv2.imwrite("mask_dirty/"+str(count)+".jpg", binary)   # Save mask
          
            # roi_2 = raw_img[y:y+h, x:x+w]
            # roi_2[binary==0] = 0 
            # cv2.imwrite("roi_2/"+str(i)+".jpg", roi_2) # Save roi without background
            
            
            #Draw the final filtered contours with bounding boxes
            cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)
            cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)

            #model inference
            class_dict={0:"landmark",1:"others"}
            os.makedirs("./runs/predict",exist_ok=True)
            for i in class_dict:
                os.makedirs("./runs/predict/"+str(i)+"/roi",exist_ok=True)
                os.makedirs("./runs/predict/"+str(i)+"/mask",exist_ok=True)
            with torch.no_grad():
                model.eval()
                roi_ = roi# READ IMAGE
                roi_ = Image.fromarray(roi_.astype('uint8'), 'RGB')
                trans_roi = pre_process(roi_)
                trans_roi = trans_roi.view([1,opts.nc,opts.img_size,opts.img_size]).cuda()
                pred = model(trans_roi)
                pred_cls = pred.argmax() #get the max score label
                print("predict result : {}".format(class_dict[int(pred_cls.cpu().numpy())]))
                print("{} after model inference".format(count))
                shutil.copy("./roi_dirty/"+str(count)+".jpg","./runs/predict/"+str(int(pred_cls.cpu().numpy()))+"/roi/")
                if save_mask:
                    shutil.copy("./mask_dirty/"+str(count)+".jpg","./runs/predict/"+str(int(pred_cls.cpu().numpy()))+"/mask/")
            count+=1
            cv2.drawContours(img, contours_poly, i%255 , cv2.CHAIN_APPROX_SIMPLE)
            
        
        #if w*h > small_size and w*h < large_size and y> (edge.shape[0]*1.0/5.0) \
        #   and float(w/h)>small_r and float(w/h)<large_r:
            # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
            # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 6)

    print("im gray {}".format(imgray.shape))

    if view_img:
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
    return count

def Get_Arrow_ROI_s(opts):
    print(opts.img_dir)
    img_path_list = glob.glob(os.path.join(opts.img_dir,"*.png"))
    print("img_path_list:{}".format(img_path_list))
    c = 1
    for img_path in img_path_list:
        print(img_path)
        c = Get_Arrow_ROI(img_path=img_path,
                        view_img=opts.view_img,
                        save_roi=opts.save_roi,
                        save_mask=opts.save_mask,
                        count=c)


if __name__=="__main__":
    # img_path = './datasets/landmark_img/Screenshot from 2023-08-10 12-50-26.png'
    # view_img = True
    # save_roi = True
    # save_mask = True
    # Get_Arrow_ROI(img_path=img_path,
    #                 view_img = view_img,
    #                 save_roi = save_roi,
    #                 save_mask = save_mask)
    Get_Arrow_ROI_s(opts)
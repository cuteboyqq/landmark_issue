import os
import shutil
import cv2
import glob
def Parse_Path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    return file,file_name

import numpy as np

def FilterImg(img_dir,
              save_dir,
              mask=True):
    os.makedirs(os.path.join(save_dir,"roi"),exist_ok=True)
    os.makedirs(os.path.join(save_dir,"mask"),exist_ok=True)
    c = 1
    img_path_list = glob.glob(os.path.join(img_dir,'*.jpg'))
    for img_path in img_path_list:
        file,file_name = Parse_Path(img_path)
        img = cv2.imread(img_path)
        h,w = img.shape[0],img.shape[1]
        print("{}:{}".format(c,img_path))
        if h*w < 40*40:
            print("too small (<40*40 pixels)")
        else:
            #roi_dir = os.path.join(save_dir,"roi")
            #shutil.copy(img_path,roi_dir)
            #print("save img")
        
            if mask:
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ## Get the mean value of gray image
                img_gray_mean = img_gray.mean()
                print(img_gray_mean)
                #var = ((img_gray - img_gray_mean) ** 2).mean()
                #std_rgb = np.sqrt(var)
                _,img_binary = cv2.threshold(img_gray,img_gray_mean+10,255,0)
                mask_dir = os.path.join(save_dir,"mask")
                cv2.imwrite(mask_dir+"/"+file,img_binary)
                print("{}:Save mask".format(c))
                equalize=True
                if equalize:
                    if img_gray_mean/1.0 +50 <=255:
                        img_tmp = int(img_gray_mean/1.0 + 50) * np.ones((int(img.shape[0]),int(img.shape[1]), 3), dtype=np.uint8)
                    else:
                        img_tmp = int(img_gray_mean/2.0 + 50) * np.ones((int(img.shape[0]),int(img.shape[1]), 3), dtype=np.uint8)
                    
                    ##lighter the landmark roi foreground
                    if img[img_binary>20].mean() < 100:
                        value = int(255 - img[img_binary>20].mean())/2.0
                    elif img[img_binary>20].mean() >=  100 and img[img_binary>20].mean() <=  180:
                        value = int(255 - img[img_binary>20].mean())/2.5
                    else:
                        value = 0

                    ##darker the landmark roi background
                    if img[img_binary<=20].mean() > 127:
                        value_bg = int(img[img_binary<=20].mean())/2.0
                    else:
                        value_bg = 10

                    ##lighter/darker the landmark roi foreground/background
                    img[img_binary>20] = img[img_binary>20] + (value,value,value)
                    img[img_binary<=20] = img[img_binary<=20] - (value_bg,value_bg,value_bg)    
                    roi_dir = os.path.join(save_dir,"roi")
                    cv2.imwrite(roi_dir+"/"+file,img)
                    print("save new img")
                #=====================================================================================
                #Because it is already the ROI image, so l do not need use contour method to get ROI
                #======================================================================================
                # contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # contours_poly = [None]*len(contours)
                # boundRect = [None]*len(contours)
                # for i, c in enumerate(contours):
                #     contours_poly[i] = cv2.approxPolyDP(c,3, True)#3
                #     boundRect[i] = cv2.boundingRect(contours_poly[i])

                # for i in range(len(contours)):
                #     x = boundRect[i][0]
                #     y = boundRect[i][1]
                #     w = boundRect[i][2]
                #     h = boundRect[i][3]
                    


        c+=1

        
        #cv2.imshow("img",img)
        #cv2.waitKey(500)
        #cv2.destroyAllWindows()
        print(img_path)

    return

if __name__=="__main__":
    img_dir = "./datasets/landmark_roi"
    save_dir = "./datasets/landmark_roi_filtered_new"
    FilterImg(img_dir,
              save_dir, 
              mask=True)
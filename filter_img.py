import os
import shutil
import cv2
import glob
import numpy as np
def Parse_Path(path):
    file = path.split(os.sep)[-1]
    file_name = file.split(".")[0]
    return file,file_name

import numpy as np

def FilterImg(img_dir,
              save_dir,
              mask=True,
              stop_sign=False,
              equalize=True,
              roi_th=None):
    os.makedirs(os.path.join(save_dir,"roi"),exist_ok=True)
    os.makedirs(os.path.join(save_dir,"mask"),exist_ok=True)
    c = 1
    img_path_list = glob.glob(os.path.join(img_dir,'*.jpg'))
    for img_path in img_path_list:
        print(img_path)
        file,file_name = Parse_Path(img_path)
        img = cv2.imread(img_path)
        h,w = img.shape[0],img.shape[1]
        print("{}:{}".format(c,img_path))
        if h*w < roi_th*roi_th or h/w < 0.10 or h/w > 10:
            print("too small (<30*30 pixels) or ratio is <0.10 or <10.0, skip this ROI")
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
                if stop_sign:
                    _,img_binary = cv2.threshold(img_gray,img_gray_mean,255,cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV
                    dilate_erode=False
                    if dilate_erode:
                        kernel = np.ones((3,3), np.uint8)
                        img_binary = cv2.dilate(img_binary, kernel, iterations = 10)
                        img_binary = cv2.erode(img_binary, kernel, iterations = 10)
                        
                else:
                    _,img_binary = cv2.threshold(img_gray,img_gray_mean+10,255,0)
                mask_dir = os.path.join(save_dir,"mask")
                #study code from https://cloud.tencent.com/developer/article/1016690
                if stop_sign:
                    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    mask_img = np.zeros(img.shape, np.uint8)
                    #cv2.drawContours(mask_img, contours, -1, (255,255,255),cv2.FILLED)
                    #cv2.drawContours(mask_img, contours, -1, (255,255,255), 3)
                    c_max = []
                    max_area = 0
                    max_cnt = 0
                    for i in range(len(contours)):
                        cnt = contours[i]
                        area = cv2.contourArea(cnt)

                        # 处理掉小的轮廓区域，这个区域的大小自己定义。
                        if(area < (h*2/3*w*2/3)):
                            c_min = []
                            c_min.append(cnt)
                            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
                            cv2.drawContours(mask_img, c_min, -1, (0,0,0), thickness=-1)
                            continue
                        #
                        c_max.append(cnt)
                    cv2.drawContours(mask_img, c_max, -1, (255, 255, 255), thickness=-1)
                    GET_BEAUTIFUL_MASK=False
                    if np.mean(mask_img) > 100:
                        GET_BEAUTIFUL_MASK=True

                    if GET_BEAUTIFUL_MASK:
                        kernel = np.ones((3,3), np.uint8)
                        mask_img = cv2.dilate(mask_img, kernel, iterations = 2)
                        cv2.imwrite(mask_dir+"/"+file,mask_img)
                        print("{}:Save mask".format(c))    
                
                if not stop_sign:
                    GET_BEAUTIFUL_MASK=True
                    if GET_BEAUTIFUL_MASK:
                        os.makedirs(mask_dir,exist_ok=True)
                        print(mask_dir)
                        print("file:{}".format(file))
                        save_mask_path = os.path.join(mask_dir,file)
                        print("save_mask_path:{}".format(save_mask_path))
                        cv2.imwrite(save_mask_path,img_binary)
                        print("{}:Save mask".format(c))
                    
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
                    if GET_BEAUTIFUL_MASK:
                        cv2.imwrite(roi_dir+"/"+file,img)
                        print("save new img")
                else:
                    img[img_binary>20] = img[img_binary>20] 
                    img[img_binary<=20] = img[img_binary<=20]  
                    roi_dir = os.path.join(save_dir,"roi")
                    if GET_BEAUTIFUL_MASK:
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
    img_dir = "/home/ali/Downloads/stop_sign"
    save_dir = "/home/ali/Downloads/stop_sign_new"
    stop_sign = True
    FilterImg(img_dir,
              save_dir, 
              mask=True,
              stop_sign=stop_sign,
              equalize=False,
              roi_th=15)
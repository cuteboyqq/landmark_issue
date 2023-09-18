import os
import glob
import shutil
import cv2
import random
import numpy as np

def Analysis_Path(path):
    img = path.split("/")[-1]
    img_name = img.split(".")[0]
    return img, img_name

def Generate_StopSign_Img(img_path=None,
                          count=None,
                          args=None):
    roi_path_list = glob.glob(os.path.join(args.roi_dirstopsign,"*.jpg"))
    IS_FAILED = False
    img_file,img_name = Analysis_Path(img_path)
    img = cv2.imread(img_path)

    #Get the random stop sign ROI first
    random_index = random.randint(0, (len(roi_path_list)-1))
    random_roi_path = roi_path_list[random_index]
    roi = cv2.imread(random_roi_path)

    dri_label = img_name + ".png"
    print(dri_label)
    dri_label_path = os.path.join(args.dri_labeldir,dri_label)
    drivable_color_map = cv2.imread(dri_label_path)

    ##Change the ROI until the ROI size > 100*100
    while(roi.shape[0]*roi.shape[1]<100*100):
        random_index = random.randint(0, (len(roi_path_list)-1))
        random_roi_path = roi_path_list[random_index]
        roi = cv2.imread(random_roi_path)

    ##Get corresponding mask
    roi_file,roi_name = Analysis_Path(random_roi_path)
    mask = roi_name + ".jpg"
    mask_path = os.path.join(args.roi_maskdirstopsign,mask)
    roi_mask = cv2.imread(mask_path)

    x = random.randint(0+int(roi.shape[1]/2.0), img.shape[1]-int(roi.shape[1]/2.0))
    y = random.randint(0+int(roi.shape[0]/2.0), img.shape[0]-int(roi.shape[0]/2.0)) 
    print(y)
    #=========================================================================================================
    label_gray = cv2.cvtColor(drivable_color_map,cv2.COLOR_BGR2GRAY)
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

    #roi = cv2.imread(roi_path)

    #roi_mask = cv2.imread(roi_mask_path)

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
        if vanish_y < carhood_y:
            y = random.randint(int(vanish_y),carhood_y-1) # Get the final  ROI y
        else:
            y = random.randint(int(img.shape[0]/2.0),int(img.shape[0]*5.0/6.0)) # Get the final  ROI y
        x = random.randint(int(w*3/10),int(w*7/10)) # Get the final  ROI x
        print("x in at background, re-random again~")
        co+=1
        if co==150:
            #IS_FAILED = True
            break
        #input()
    # final_x_stop_sign = random.randint(int(0),int(w-1)) # Get the final  ROI x
    # while(label_mask[y][final_x_stop_sign]!=0): #for stop sign
    #     final_x_stop_sign = random.randint(int(0),int(w-1)) # Get the final  ROI x
    #     print("[Stop sign] x not in at background, re-random again~")
    #     co+=1
    #     if co==150:
    #         IS_FAILED = True
    #         break
    left_background_boundary = 0
    while(label_mask[y][left_background_boundary]==0):
        if left_background_boundary+1<img.shape[1]:
            left_background_boundary+=1
        else:
            break

    print("left_background_boundary:{}".format(left_background_boundary))

    right_background_boundary = img.shape[1] - 1
    while(label_mask[y][right_background_boundary]==0):
        if right_background_boundary-1>0:
            right_background_boundary-=1
        else:
            break
        

    print("right_background_boundary:{}".format(right_background_boundary))

    #input()
    #============find the middle of drivable area=========================
    left_line_point_x = 0
    search_x = x
    #(r,g,b) = label[y][x]
    print(drivable_color_map[y][x])
    while(search_x>0):
        if drivable_color_map[y][search_x][0]== drivable_color_map[y][x][0]:
            search_x -=1
        elif not drivable_color_map[y][search_x][0]== drivable_color_map[y][x][0] :
            left_line_point_x = search_x
            break

    print("left_line_point_x:{}".format(left_line_point_x))
    search_x = x
    right_line_point_x = left_line_point_x + 2
    while(search_x<label_mask.shape[1]):
        if drivable_color_map[y][search_x][0]==drivable_color_map[y][x][0]:
            search_x +=1
        elif not drivable_color_map[y][search_x][0]==drivable_color_map[y][x][0] :
            right_line_point_x = search_x
            break
        elif drivable_color_map[y][search_x][0]==drivable_color_map[y][x][0] and search_x==label_mask.shape[1]*5.0/6.0:
            right_line_point_x = search_x

    print("right_line_point_x:{}".format(right_line_point_x))


    final_x = int((left_line_point_x + right_line_point_x )/2.0)
    print("final_x:{}".format(final_x))
    print("final_y:{}".format(y))
    road_width = abs(right_line_point_x - left_line_point_x)
    print("road_width = {}".format(road_width))


    roi_w, roi_h = roi.shape[1], roi.shape[0]
    #Set landmark width (initial setting)
    final_roi_w = int(road_width * 0.30) #Get the final ROI width



    left_width = left_background_boundary
    right_width = img.shape[1] - right_background_boundary
    if final_roi_w >= left_width or final_roi_w >= right_width:
        final_roi_w = int(min(left_width,right_width) * 0.30)

    resize_ratio = float(final_roi_w/roi_w)
    final_roi_h = int(roi.shape[0]*resize_ratio)
    if final_roi_w==0 or final_roi_h==0:
        final_roi_w = 50
        final_roi_h = 50

    #print("final_roi_w:{}".format(final_roi_w))
    #print("final_roi_w:{}".format(final_roi_h))

    draw_boundary=False
    if draw_boundary:
        cv2.line(img,(left_background_boundary,0),(left_background_boundary,img.shape[0]-1),(255,0,0),4)
        cv2.line(img,(right_background_boundary,0),(right_background_boundary,img.shape[0]-1),(255,0,0),4)
        cv2.line(img,(left_background_boundary,y),(right_background_boundary,y),(255,0,0),4)
    random_left_right = random.randint(0,1)
    if random_left_right==0:
        final_x_stop_sign = int(left_background_boundary)
    else:
        final_x_stop_sign = int(right_background_boundary)
    #=========================================================================================================
    method = random.randint(0,1) #Random method opencv/mask
    if method==0:
        USE_OPENCV=True
    else:
        USE_OPENCV=False

    #Set by User
    if not args.use_mask:
        USE_OPENCV=True
    elif not args.opencv:
        USE_OPENCV=False

    ## Resize the Stopsign ROI & ROI mask
    ##Stopsign width/height size is range from 60~180
    if final_roi_w<180 and final_roi_h<180 and\
       final_roi_w>=60 and final_roi_h>=60:
        roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
    elif final_roi_w<60 and final_roi_h<60:
        roi_resize = cv2.resize(roi,(60,60),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(60,60),interpolation=cv2.INTER_NEAREST)
        final_roi_h=60
        final_roi_w=60
    else:
        roi_resize = cv2.resize(roi,(180,180),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(180,180),interpolation=cv2.INTER_NEAREST)
        final_roi_h=180
        final_roi_w=180

    #print("roi_resize:{}".format(roi_resize.shape))
    #print("roi_mask:{}".format(roi_mask.shape))
    ##Get ROI coordinate y
    ## SopSign is about 2.0 meters tall, so y is smaller
    if roi_resize.shape[0]<100:
        y = y - roi_resize.shape[0] * 3 if y - roi_resize.shape[0] * 3 > 0 else y - roi_resize.shape[0]
    else:
        y = y - roi_resize.shape[0] * 1

    ## Generate image with Stop sign
    if USE_OPENCV:
        mask = 255 * np.ones(roi_resize.shape, roi_resize.dtype)
        center = (final_x_stop_sign,y) #wrong location
        #for i in range(100):
        try:
            output = cv2.seamlessClone(roi_resize, img, mask, center, cv2.NORMAL_CLONE)   #MIXED_CLONE
        except:
            IS_FAILED=True
            pass
    else: # Use Mask method
        roi_tmp = np.zeros(roi_resize.shape, dtype=np.uint8)
        #print("roi_tmp:{}".format(roi_tmp.shape))
        ## Pre-process the coordinate 
        h_r = int(final_roi_h)
        #print("h_r = {}".format(h_r))
        h_add = 0
        if h_r%2!=0:
            h_add = 1

        w_r = int(final_roi_w)
        #print("w_r = {}".format(w_r))
        w_add = 0
        if w_r%2!=0:
            w_add = 1


        ## Get the image ROI at coordinate (x,y)= (final_x_stop_sign,y) w=final_roi_w h=final_roi_h
        x_c = final_x_stop_sign
        y_c = y
        
        if x_c-int(final_roi_w/2.0)<=0:
            x_c = int(final_roi_w/2.0) + 1
        
        if x_c+int(final_roi_w/2.0)+w_add >= img.shape[1]:
            x_c = x_c - (int(final_roi_w/2.0)+w_add+1)

        if y_c-int(final_roi_h/2.0)<=0:
            y_c =  int(final_roi_h/2.0) + 1
        
        if y_c+int(final_roi_h/2.0)+h_add>=img.shape[0]:
            y_c = y_c-(int(final_roi_h/2.0)+h_add+1)

        ## Update ROI center x,y
        final_x_stop_sign = x_c
        y = y_c
        
        #print("{}, {}".format(x_c,y_c))
        img_roi = img[y_c-int(final_roi_h/2.0):y_c+int(final_roi_h/2.0)+h_add, x_c-int(final_roi_w/2.0):x_c+int(final_roi_w/2.0)+w_add]
        #print("img_roi:{}".format(img_roi.shape))
        #input()
        ##Process the stop sign ROI, remove background by mask
        roi_tmp[roi_mask>20] = roi_resize[roi_mask>20] #Foreground using stop sign 
        roi_tmp[roi_mask<=20] = img_roi[roi_mask<=20] #Background using image background

        ##Put the stop sign ROI into bdd100k image       
        img[y_c-int(final_roi_h/2.0):y_c+int(final_roi_h/2.0)+h_add, x_c-int(final_roi_w/2.0):x_c+int(final_roi_w/2.0)+w_add] = roi_tmp



    if args.save_txt and IS_FAILED==False:
        print("args.save_txt : {}".format(args.save_txt))

        label_txt = img_name + ".txt"
        label_txt_path=os.path.join(args.label_dir,label_txt)
        if os.path.exists(label_txt_path):
            #input()
            label = str(10) #stop sign label = 10 replace the traffic sign
            x_cor = final_x_stop_sign
            x = str( int(float( (x_cor) / img.shape[1] )*1000000)/1000000 ) 
            y = str( int(float( (y    ) / img.shape[0] )*1000000)/1000000 )
            w = str( int((roi_resize.shape[1]/img.shape[1])*1000000)/1000000)
            h = str( int((roi_resize.shape[0]/img.shape[0])*1000000)/1000000)
            line = label + " " + x + " " + y + " " + w + " " +h
            
            
            save_label_txt_dir = os.path.join(args.save_dir,"labels")
            print(save_label_txt_dir)
            #input()
            os.makedirs(save_label_txt_dir,exist_ok=True)

            shutil.copy(label_txt_path,save_label_txt_dir)

            save_label_txt_path = save_label_txt_dir + "/" + label_txt
            
            f = open(save_label_txt_path,'a')

            with open(label_txt_path,'r') as f_ori: #original label from bdd100k.txt 
                lines_ori = f_ori.readlines()
                for line_ori in lines_ori:
                    if line_ori.split(" ")[0] != "9":
                        f.write(line_ori)
                        f.write("\n")
            f.close()           

            with open(save_label_txt_path,'a') as f:
                f.write(line)
                f.write("\n")
            f.close()
        else:
            print("No label.txt")
            IS_FAILED=True

    if args.show_img:
        
        if USE_OPENCV:
            try:
                cv2.imshow("output",output)
            except:
                pass
        else:
            cv2.imshow("img",img)

        #cv2.imshow("drivable",drivable)
        #cv2.imshow("roi",roi)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
    if args.save_img and IS_FAILED==False:
        try:
            os.makedirs(args.save_dir + "/images",exist_ok=True)
            save_img_path = os.path.join(args.save_dir,"images",img_file)
            if USE_OPENCV:
                cv2.imwrite(save_img_path,output)
            else:
                cv2.imwrite(save_img_path,img)
            print("=======================================count:{}".format(count))
            count+=1
        except:
            pass

    return count
def Generate_StopSign_Imgs(args=None):
    IS_FAILED=False
    img_path_list = glob.glob(os.path.join(args.img_dir,"*.jpg"))
    #roi_path_list = glob.glob(os.path.join(args.roi_dirstopsign,"*.jpg"))
    count=1
    for img_path in img_path_list:
        count = Generate_StopSign_Img(img_path,count,args)

        if count>args.num_img:
            break



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    parser.add_argument('-drilabeldir','--dri-labeldir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    parser.add_argument('-roidirstopsign','--roi-dirstopsign',help='roi dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v2/roi")
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/stopsign_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-numimg','--num-img',type=int,default=20000,help='number of generate fake landmark images')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='enable(True)/disable(False) mask method to generate landmark or not')
    parser.add_argument('-roimaskdirstopsign','--roi-maskdirstopsign',help='roi mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v2/mask")
    parser.add_argument('-opencv','--opencv',type=bool,default=True,help='enable(True)/disable(False) opencv method to generate landmark or not')
    return parser.parse_args()  
  
if __name__=="__main__":
    args = get_args()
    FINISHED = Generate_StopSign_Imgs(args)
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

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #assert iou >= 0.0
    #assert iou <= 1.0
    return iou

def xywh_to_xyxy(xywh):
    x,y,w,h = xywh[0],xywh[1],xywh[2],xywh[3]
    x1 = int(x) - int(w/2.0) -1
    y1 = int(y) - int(h/2.0) -1
    x2 = int(x) + int(w/2.0) +1
    y2 = int(y) + int(h/2.0) +1
    xyxy = [x1,y1,x2,y2]
    return xyxy

def Generate_StopSign_Img(img_path=None,count=1,args=None):
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
    real_y = random.randint(int(img.shape[0]/5.0),carhood_y-1) # Get the final  ROI y
   
    HVAE_LABEL_TXT=True
    ## Get the loaction (x,y) to put stop sign ROI into bdd100k image
    X_TH=100
    Y_TH=100
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
    print("final_y:{}".format(real_y))
    road_width = abs(right_line_point_x - left_line_point_x)
    print("road_width = {}".format(road_width))


    roi_w, roi_h = roi.shape[1], roi.shape[0]
    #Set landmark width (initial setting)
    random_ratio = random.randint(3,20)
    final_roi_w = int(road_width * random_ratio * 0.1) #Get the final ROI width



    left_width = left_background_boundary
    right_width = img.shape[1] - right_background_boundary
    if real_y > vanish_y:
        if final_roi_w >= left_width or final_roi_w >= right_width:
            if roi_w>60:
                final_roi_w = random.randint( int(min(left_width,right_width) * args.w_minra), int(min(left_width,right_width) * args.w_maxra) )
            elif roi_w<=60:
                final_roi_w = random.randint( int(min(left_width,right_width) * args.w_minra * 2.0), int(min(left_width,right_width) * 1.0) )
    else:
        #final_roi_w = random.randint( int(roi.shape[1] * args.w_minra), int(roi.shape[1] * 1.0) )
        print("real_y <= vanish_y")
        ## l want small stop sign inthe middle of image size si about : 20*20~40*40
        final_roi_w = random.randint(20,40)
        

    resize_ratio = float(final_roi_w/roi_w)
    final_roi_h = int(roi.shape[0]*resize_ratio)


    ## open label.txt
    label_txt = img_name + ".txt"
    label_txt_path=os.path.join(args.label_dir,label_txt)
    xywh_stop_sign = [x,real_y,final_roi_w,final_roi_h]
    print("xywh_stop_sign:{}".format(xywh_stop_sign))
    xyxy_stop_sign = xywh_to_xyxy(xywh_stop_sign)
    
    OVERLAPPED=False
    if os.path.exists(label_txt_path):
        HVAE_LABEL_TXT=True
        with open(label_txt_path,'r') as f:
            ## Start parsing label.txt
            lines = f.readlines()
            for line in lines:
                label_bdd=int(line.split(" ")[0])
                x_bdd=float(line.split(" ")[1])*img.shape[1]
                y_bdd=float(line.split(" ")[2])*img.shape[0]
                w_bdd=float(line.split(" ")[3])*img.shape[1]
                h_bdd=float(line.split(" ")[4])*img.shape[0]
                
                bb_bdd_xywh = [x_bdd,y_bdd,w_bdd,h_bdd]
                bb_bdd_xyxy = xywh_to_xyxy(bb_bdd_xywh)    
                
                bb_stopsign = {"x1":xyxy_stop_sign[0],
                               "x2":xyxy_stop_sign[2],
                               "y1":xyxy_stop_sign[1],
                               "y2":xyxy_stop_sign[3]}
                
                bb_bdd100k = {"x1":bb_bdd_xyxy[0],
                              "x2":bb_bdd_xyxy[2],
                               "y1":bb_bdd_xyxy[1],
                               "y2":bb_bdd_xyxy[3]}
                
                print("bb_stopsign:{}".format(bb_stopsign))
                print("bb_bdd100k:{}".format(bb_bdd100k))

                iou = get_iou(bb_stopsign,bb_bdd100k)
                if iou > 0.30:
                    OVERLAPPED=True
                    IS_FAILED=True
                    print("iou={}".format(iou))
                    #input()
                else:
                    print("iou={}".format(iou))
                    #input()
                    
                ## if (x,y) is close to another label BB, do not use
                #if abs(x_bdd-x)<X_TH and abs(y_bdd-y)<Y_TH:
                    #overlap_with_car=True
    else:
        HVAE_LABEL_TXT=False


    if final_roi_w==0 or final_roi_h==0:
        print("final_roi_w or final_roi_h is 0")
        final_roi_w = 20
        resize_ratio = float(final_roi_w/roi_w)
        final_roi_h = int(roi.shape[0]*resize_ratio)
        #input()

    print("final_roi_w:{}".format(final_roi_w))
    print("final_roi_w:{}".format(final_roi_h))
    #input()
    draw_boundary=False
    if draw_boundary:
        cv2.line(img,(left_background_boundary,0),(left_background_boundary,img.shape[0]-1),(255,0,0),4)
        cv2.line(img,(right_background_boundary,0),(right_background_boundary,img.shape[0]-1),(255,0,0),4)
        cv2.line(img,(left_background_boundary,y),(right_background_boundary,y),(255,0,0),4)

    ## Random Stop Sign x at random left/right side
    random_left_right = random.randint(0,1)
    if real_y >= vanish_y:
        if random_left_right==0:
            final_x_stop_sign = random.randint(0,int(left_background_boundary))
        else:
            final_x_stop_sign = random.randint(int(right_background_boundary), int(img.shape[1]-1))
    else:
        final_x_stop_sign = random.randint(int(img.shape[1]*4.0/10.0),int(img.shape[1]*6.0/10.0))
        #final_x_stop_sign = int(img.shape[1]/2.0)
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
    #print("USE_OPENCV : {}".format(USE_OPENCV))
    #input()
    ##Resize the Stopsign ROI & ROI mask
    if final_roi_w<args.roi_maxwidth and final_roi_w>=20 and final_roi_h<args.roi_maxwidth and final_roi_h>=20:
        roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
    elif final_roi_w>=args.roi_maxwidth and final_roi_h>=args.roi_maxwidth:
        roi_resize = cv2.resize(roi,(args.roi_maxwidth,args.roi_maxwidth),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(args.roi_maxwidth,args.roi_maxwidth),interpolation=cv2.INTER_NEAREST)
        final_roi_h=args.roi_maxwidth
        final_roi_w=args.roi_maxwidth
    else:
        final_roi_w=20
        resize_ratio = float(final_roi_w/roi_w)
        final_roi_h = int(roi.shape[0]*resize_ratio)
        roi_resize = cv2.resize(roi,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
        roi_mask = cv2.resize(roi_mask,(final_roi_w,final_roi_h),interpolation=cv2.INTER_NEAREST)
        

    print("roi_resize:{}".format(roi_resize.shape))
    print("roi_mask:{}".format(roi_mask.shape))
    ##Get ROI coordinate y
    if roi_resize.shape[0]<100:
        y = y - roi_resize.shape[0] * 2 if y - roi_resize.shape[0] * 2 > 0 else y - roi_resize.shape[0]
    else:
        y = y - roi_resize.shape[0] * 1

    ## Generate image with Stop sign
    if USE_OPENCV:
        mask = 255 * np.ones(roi_resize.shape, roi_resize.dtype)
        center = (final_x_stop_sign,real_y) #wrong location
        #for i in range(100):
        try:
            output = cv2.seamlessClone(roi_resize, img, mask, center, cv2.NORMAL_CLONE)   #MIXED_CLONE
        except:
            IS_FAILED=True
            pass
    else: # Use Mask method
        roi_tmp = np.zeros(roi_resize.shape, dtype=np.uint8)
        print("roi_tmp:{}".format(roi_tmp.shape))
        ##Pre-process the coordinate 
        h_r = int(final_roi_h)
        print("h_r = {}".format(h_r))
        h_add = 0
        if h_r%2!=0:
            h_add = 1

        w_r = int(final_roi_w)
        print("w_r = {}".format(w_r))
        w_add = 0
        if w_r%2!=0:
            w_add = 1


        ##Get the image ROI at coordinate (x,y)= (final_x_stop_sign,y) w=final_roi_w h=final_roi_h
        x_c = final_x_stop_sign
        y_c = real_y
        

        ## keep the road sign at the left/right side of the image
        # if x_c > int(img.shape[1] * 0.28) and x_c < int(img.shape[1] * 0.50):
        #     x_c = int(img.shape[1] * 0.28)

        # elif x_c < int(img.shape[1] * 0.72) and x_c > int(img.shape[1] * 0.50):
        #     x_c = int(img.shape[1] * 0.72)
            

        ## keep the road sign in the images
        if x_c-int(final_roi_w/2.0)<=0:
            x_c = int(final_roi_w/2.0) + 1
        
        if x_c+int(final_roi_w/2.0)+w_add >= img.shape[1]:
            x_c = x_c - (int(final_roi_w/2.0)+w_add+1)

        if y_c-int(final_roi_h/2.0)<=0:
            y_c =  int(final_roi_h/2.0) + 1
        
        if y_c+int(final_roi_h/2.0)+h_add>=img.shape[0]:
            y_c = y_c-(int(final_roi_h/2.0)+h_add+1)

        #update x,y
        final_x_stop_sign = x_c
        y = y_c
        
        print("{}, {}".format(x_c,y_c))
        img_roi = img[y_c-int(final_roi_h/2.0):y_c+int(final_roi_h/2.0)+h_add, x_c-int(final_roi_w/2.0):x_c+int(final_roi_w/2.0)+w_add]
        print("img_roi:{}".format(img_roi.shape))
        #input()
        ##Process the stop sign ROI, remove background by mask
        roi_tmp[roi_mask>20] = roi_resize[roi_mask>20] #Foreground using stop sign 
        roi_tmp[roi_mask<=20] = img_roi[roi_mask<=20] #Background using image background

        ##Put the stop sign ROI into bdd100k image       
        img[y_c-int(final_roi_h/2.0):y_c+int(final_roi_h/2.0)+h_add, x_c-int(final_roi_w/2.0):x_c+int(final_roi_w/2.0)+w_add] = roi_tmp


    TXT_SAVED = False
    # Generate_StopSign_Img
    if args.save_txt and IS_FAILED==False:
        print("args.save_txt : {}".format(args.save_txt))

        label_txt = img_name + ".txt"
        label_txt_path=os.path.join(args.label_dir,label_txt)
        if os.path.exists(label_txt_path):
            #input()
            REMOVE_TRAFFIC_SIGN_LABLE=True
            if REMOVE_TRAFFIC_SIGN_LABLE:
                label = str(9) ## No :Stop sign label is 9, replace the traffic sign label , Yes: Add new label 10 for the stop sign~~
            else:
                label = str(10)
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

            with open(save_label_txt_path,'w') as f:
                ## remove the traffic sign label
                with open(label_txt_path,'r') as bdd100k_f:
                    bdd100k_lines = bdd100k_f.readlines()
                    for bdd100k_line in bdd100k_lines:
                        if bdd100k_line.split(" ")[0]!="9": #if not traffic sign label
                            f.write(bdd100k_line)
            f.close()
            ## Add the Stop sign label
            with open(save_label_txt_path,'a') as f:
                f.write(line)
                #f.write("\n")
            f.close()
            TXT_SAVED=True
        else:
            print("No label.txt")
            IS_FAILED=True
            TXT_SAVED=False

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

    IMG_SAVED=False
    if args.save_img and IS_FAILED==False:
        try:
            os.makedirs(args.save_dir + "/images",exist_ok=True)
            save_img_path = os.path.join(args.save_dir,"images",img_file)
            if USE_OPENCV:
                cv2.imwrite(save_img_path,output)
            else:
                cv2.imwrite(save_img_path,img)
            print("=================================count:{}".format(count))
            count+=1
            IMG_SAVED=True
           
        except:
            pass

    if IMG_SAVED and TXT_SAVED:
        return count, save_img_path, save_label_txt_path
    else:
        return count, None, None


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
                          use_opencv_ratio=0.25,
                          use_stoptsign_dataset=True,
                          Saved_StopSign_Txt_Path=None,
                          count_final=None):
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
            return IS_FAILED,count_final
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
        return IS_FAILED,count_final
    
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
                return IS_FAILED,count_final
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
                            return IS_FAILED,count_final
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
                return IS_FAILED,count_final
            
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
                return IS_FAILED,count_final
            
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
                    return IS_FAILED,count_final
            else:
                output1 = img
                IS_FAILED = True
                return IS_FAILED,count_final
            if show_img:
                cv2.imshow("output1",output1) #img
            if save_landmark_img and not IS_FAILED:

                #Now we have information of ROI coordinate : x,y,w,h , so we can save it into filename.txt of yolo format
                if save_txt:
                    os.makedirs("./fake_landmark_image_test/label",exist_ok=True)
                    image, img_name = Analysis_path(img_path)
                    REMOVE_TRAFFIC_SIGN_LABLE = True
                    if REMOVE_TRAFFIC_SIGN_LABLE:
                        landmark_label = 10 # 10 is for lane marking  used when remove traffic sign label~~
                    else:
                        landmark_label = 11
                    x = float(int((final_x/img.shape[1])*1000000)/1000000)
                    y = float(int((y/img.shape[0])*1000000)/1000000)
                    w = float(int((final_roi_w/img.shape[1])*1000000)/1000000)
                    h = float(int((final_roi_h/img.shape[0])*1000000)/1000000)
                
                    if use_stoptsign_dataset:
                        use_bdd100k_dataset = False

                    if use_bdd100k_dataset:
                        try:
                            #print("use bdd100k dataset !!")
                            #input()
                            f = open("/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train/"+img_name+".txt",'r')
                            
                            f2 = open("./fake_landmark_image_test/label/"+img_name+".txt",'w')
                            f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                            f2.write("\n")
                            for line in f.readlines():
                                f2.write(line)

                            f2.close()
                            f.close()
                            count_final+=1
                        except:
                            print("Pass saving labels.txt,No bdd100k label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED,count_final
                    elif use_stoptsign_dataset:
                        try:
                            #f = open("/home/ali/Projects/GitHub_Code/ali/landmark_issue/stopsign_images/labels/"+img_name+".txt",'r')
                            #print("use stop sign images")
                            #print(Saved_StopSign_Txt_Path)
                            #input()
                            f = open(Saved_StopSign_Txt_Path,'r')
                            #print("open success!!")
                            #input()
                            f2 = open("./fake_landmark_image_test/label/"+img_name+".txt",'a')
                            f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                            f2.write("\n")
                            # for line in f.readlines():
                            #     f2.write(line)
                            lines = f.readlines()
                            for line in lines:
                                #print(line)
                                f2.write(line)

                            f2.close()
                            f.close()
                            count_final+=1
                            #input()
                        except:
                            print("Pass saving labels.txt,No StopSign label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED,count_final
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
            return IS_FAILED,count_final
        
        if y> (vanish_y)+ abs(carhood_y-vanish_y)/10.0 and y<carhood_y-1:
            print("case 1 ")
            try:
                roi_l = cv2.resize(roi,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
                roi_mask = cv2.resize(roi_mask,(int(w_r*resize_ratio),int(h_r*resize_ratio_h)),interpolation=cv2.INTER_NEAREST)
            except:
                IS_FAILED=True
                return IS_FAILED,count_final
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
                return IS_FAILED,count_final
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
                return IS_FAILED,count_final
        
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
                            return IS_FAILED,count_final
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
                    return IS_FAILED,count_final
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
                    landmark_label = 11
                    x = float(int((final_x/img.shape[1])*1000000)/1000000)
                    y = float(int((y/img.shape[0])*1000000)/1000000)
                    w = float(int((final_roi_w/img.shape[1])*1000000)/1000000)
                    h = float(int((final_roi_h/img.shape[0])*1000000)/1000000)
                    if use_stoptsign_dataset:
                        use_bdd100k_dataset = False

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
                            count_final+=1
                        except:
                            print("Pass saving labels.txt,No bdd100k label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED,count_final
                    elif use_stoptsign_dataset:
                        try:
                            #f = open("/home/ali/Projects/GitHub_Code/ali/landmark_issue/stopsign_images/labels/"+img_name+".txt",'r')
                            #print("use stop sign images")
                            #print(Saved_StopSign_Txt_Path)
                            #input()
                            f = open(Saved_StopSign_Txt_Path,'r')
                            #print("open success!!")
                            #input()
                            f2 = open("./fake_landmark_image_test/label/"+img_name+".txt",'a')
                            f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                            f2.write("\n")
                            # for line in f.readlines():
                            #     f2.write(line)
                            lines = f.readlines()
                            for line in lines:
                                #print(line)
                                f2.write(line)

                            f2.close()
                            f.close()
                            #input()
                            count_final+=1
                        except:
                            print("Pass saving labels.txt,No StopSign label.txt~ ")
                            IS_FAILED=True
                            return IS_FAILED,count_final
                    else:
                        f2 = open("./runs/predict/label/"+img_name+".txt",'w')
                        f2.write(str(landmark_label)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
                        f2.close()


                image, img_name = Analysis_path(img_path)
                landmark_img = image
                label = img_name + ".png"
                image_dir = os.path.join("./fake_landmark_image_test","images")
                os.makedirs(image_dir,exist_ok=True)
                cv2.imwrite("./fake_landmark_image_test/images/"+landmark_img,img)

                if save_mask:
                    label_dir = os.path.join("./fake_landmark_image_test","masks")
                    os.makedirs(label_dir,exist_ok=True)
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
    
    return IS_FAILED,count_final



def Generate_LaneMarking_StopSign_Imgs(args_stopsign=None,
                                    args_roadsign=None):
    IS_FAILED=False
    img_path_list = glob.glob(os.path.join(args_stopsign.img_dir,"*.jpg"))
    #roi_path_list = glob.glob(os.path.join(args.roi_dirstopsign,"*.jpg"))
    count = 1
    count_final = 1
    roi_path_list = glob.glob(os.path.join(args_roadsign.roi_dir,"*.jpg"))
    for img_path in img_path_list:
        ## Step 1 : Generate Stop Sign Image First
        count, Saved_StopSign_Img_Path, Saved_StopSign_Txt_Path = Generate_StopSign_Img(img_path,count,args_stopsign)

        ## Step 2 : Generate the LaneMarking Image
        if not Saved_StopSign_Img_Path==None and not Saved_StopSign_Txt_Path==None:
            ## Init paremeters for lanemark
            img, img_name = Analysis_path(Saved_StopSign_Img_Path)
            label = img_name + ".png"
            label_path = os.path.join(args_roadsign.dri_labeldir,label)
            label_train_path = os.path.join(args_roadsign.dri_labeldirtrain,label)
            #print("label_train_path:{}".format(label_train_path))
            line_label_path = os.path.join(args_roadsign.line_labeldir,label)
            print(label_path)
            #random choose landmark mask
            selected_roi_path = roi_path_list[random.randint(0,len(roi_path_list)-1)]
            r, r_name = Analysis_path(selected_roi_path)
            selected_mask_path = os.path.join(args_roadsign.roi_maskdir,r)

            IS_FAILED = True
            count_lanemarking = 1
            while(IS_FAILED and count_lanemarking!=10):
                #random choose landmark mask
                selected_roi_path = roi_path_list[random.randint(0,len(roi_path_list)-1)]
                r, r_name = Analysis_path(selected_roi_path)
                selected_mask_path = os.path.join(args_roadsign.roi_maskdir,r)

                IS_FAILED,count_final = Generate_Landmark_Img(img_path=Saved_StopSign_Img_Path,
                                roi_path=selected_roi_path,
                                roi_mask_path=selected_mask_path,
                                label_path=label_path,
                                label_train_path=label_train_path,
                                line_label_path=line_label_path,
                                save_landmark_img=True, #args_roadsign.save_img
                                save_colormap=False,#args_roadsign.save_colormap
                                save_mask=False,#args_roadsign.save_mask,
                                save_txt=True,#args_roadsign.save_txt
                                show_img=args_roadsign.show,
                                use_mask=args_roadsign.use_mask,
                                use_opencv_ratio=args_roadsign.use_opencvratio,
                                use_stoptsign_dataset=True,#args_roadsign.use_stopsigndataset
                                Saved_StopSign_Txt_Path=Saved_StopSign_Txt_Path,
                                count_final=count_final
                                )
                                
                count_lanemarking+=1
        
        if count_final>args_stopsign.num_img:
            FINISHED=True
            return FINISHED

def get_args_StopSign():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    parser.add_argument('-drilabeldir','--dri-labeldir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    # For StopSign parameter
    parser.add_argument('-roidirstopsign','--roi-dirstopsign',help='roi dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/roi")
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/stopsign_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-numimg','--num-img',type=int,default=25000,help='number of generate fake landmark images')
    parser.add_argument('-roimaxwidth','--roi-maxwidth',type=int,default=400,help='max width of stop sign ROI')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='enable(True)/disable(False) mask method to generate landmark or not')
    parser.add_argument('-roimaskdirstopsign','--roi-maskdirstopsign',help='roi mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/mask")
    parser.add_argument('-opencv','--opencv',type=bool,default=False,help='enable(True)/disable(False) opencv method to generate landmark or not')
    parser.add_argument('-wminra','--w-minra',type=float,default=0.05,help='min stop sign roi width ratio (min_th * raod width)')
    parser.add_argument('-wmaxra','--w-maxra',type=float,default=1.50,help='max stop sign roi width ratio (max_th * road_width)')
    return parser.parse_args() 

def get_args_RoadSign():
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
    parser.add_argument('-numimg','--num-img',type=int,default=25000,help='number of generate fake landmark images')
    parser.add_argument('-useopencvratio','--use-opencvratio',type=float,default=0.50,help='ratio of using opencv method to generate landmark images')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='use mask method to generate landmark or not')
    parser.add_argument('-show','--show',action='store_true',help='show images result')
    parser.add_argument('-usestopsigndataset','--use-stopsigndataset',type=bool,default=True,help='use stop sign label.txt dataset')
    return parser.parse_args()    

def get_args_Pedestrian_Motorcycle_Rider():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/pedestrain_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')

    return parser.parse_args()


if __name__=="__main__":
    args_stopsign                   = get_args_StopSign()
    args_roadsign                   = get_args_RoadSign()
    args_pedestian_motorcycle_rider = get_args_Pedestrian_Motorcycle_Rider()
    FINISHED = Generate_LaneMarking_StopSign_Imgs(args_stopsign,
                                               args_roadsign)
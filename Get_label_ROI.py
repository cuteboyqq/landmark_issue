import os
import glob
import cv2
import shutil

def Analysis_Path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    return file, file_name

def xywh2xyxy(line,normalize=True):
    
    return NotImplemented

def Analysis_yolo_txt(img_path=None,
                      label_path=None,
                      img_h=None,
                      img_w=None,
                      save_img=False,
                      args=None):
    wanted_label_xywh_list = []
    #wanted_label_xywh_list put [[int x,int y,int w,int h],...]
    if os.path.exists(label_path):
            with open(label_path,"r") as l_f:
                lines = l_f.readlines()
                for line in lines:
                    #print(line)
                    # 23    0.185977 0.901608 0.206297 0.129554
                    # cls   x           y       w           h 
                    #xyxy = xywh2xyxy(line)

                    ##Un-normalize
                    parse_line = line.split(" ")
                    label = parse_line[0]
                    #print(label)
                    #print(args.wanted_label)
                    if int(label) == args.wanted_label:
                        x = int(float(parse_line[1])*img_w - (float(parse_line[3])*img_w / 2.0)) # un-normalize top left x
                        y = int(float(parse_line[2])*img_h - (float(parse_line[4])*img_h / 2.0)) # un-normalize top left y
                        w = int(float(parse_line[3])*img_w) # un-normalize w
                        h = int(float(parse_line[4])*img_h) # un-normalize h
                        wanted_label_xywh_list.append([x,y,w,h])
                        if save_img:
                            save_img_dir = os.path.join(args.save_dir,"images")
                            os.makedirs(save_img_dir,exist_ok=True)
                            shutil.copy(img_path,save_img_dir)

            l_f.close()
    else:
        print("[Error] label_path not found !")

    return wanted_label_xywh_list


def Get_Secific_Label_ROI_Imgs(args=None):
    
    img_path_list = glob.glob(os.path.join(args.img_dir,"*.jpg"))
    c=1
    c_roi = 1
    for img_path in img_path_list:
        #print("{}:{}".format(c,img_path))
        img, img_name = Analysis_Path(img_path)

        img = cv2.imread(img_path)
        img_h,img_w = img.shape[0],img.shape[1]
        
        label = img_name + ".txt"
        label_path = os.path.join(args.label_dir,label)#Get label.txt path
        #print("label : {}".format(label))
        #print(args.label_dir)
        #print(label_path)
        wanted_label_xywh_list = Analysis_yolo_txt(img_path=img_path,
                                    label_path=label_path,
                                    img_h=img_h,
                                    img_w=img_w,
                                    save_img=True,
                                    args=args)
        #print(wanted_label_xywh_list)

        for i in range(len(wanted_label_xywh_list)):
            xywh = wanted_label_xywh_list[i]
            x = xywh[0]
            y = xywh[1]
            w = xywh[2]
            h = xywh[3]
            ##Crop ROI
            img_roi = img[y:y+h,x:x+w]
            save_roi_dir = os.path.join(args.save_dir,str(args.wanted_label))
            os.makedirs(save_roi_dir,exist_ok=True)
            cv2.imwrite(save_roi_dir + "/" + str(c_roi) + ".jpg",img_roi)
            c_roi+=1
            print(c_roi)
        if args.show_img:
            if len(wanted_label_xywh_list)>0:
                cv2.imshow("img",img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        c+=1
    

    #return NotImplemented


def get_args():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("-imgdir","--img-dir",help="image directory",default="/home/ali/Projects/GitHub_Code/YOLO/datasets/coco/images/train2017")
    parser.add_argument("-labeldir","--label-dir",help="label directory",default="/home/ali/Projects/GitHub_Code/YOLO/datasets/coco/labels/train2017")
    parser.add_argument("-wantedlabel","--wanted-label",type = int, help="wanted label (int cls)",default=11)
    parser.add_argument("-savedir","--save-dir",help="save roi directory",default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/stop_sign_20230829")
    parser.add_argument("-showimg","--show-img",help="show wanted label images",action='store_true')
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    Get_Secific_Label_ROI_Imgs(args)
    #===coco 2017 labels==================================
    # person
    # bicycle
    # car
    # motorcycle
    # airplane
    # bus
    # train
    # truck
    # boat
    # traffic light
    # fire hydrant
    # stop sign
    # parking meter
    # bench
    # bird
    # cat
    # dog
    # horse
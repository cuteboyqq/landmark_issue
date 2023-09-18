import os
import glob
import shutil
import cv2
import random
import numpy as np

#https://www.learncodewithmike.com/2020/01/python-inheritance.html



class BaseDatasets:
    """
    BaseDatasets

    A base class for create dataset for cpoy roi into image.

    Attributes:
        args (SimpleNamespace): Configuration for the dataset.
        save_dir (Path): Directory to save results.
        img_dir (Path): Directory to the images.
        label_dir (Path): Directory to the label.
        roi_dir (Path): Directory to the roi.
        roi_mask_dir (Path): Directory to the mask roi
        drivable_area_label_dir (Path): Directory to the drivable area label.
        roi_label (int): label of the roi.
        num_of_roi_into_image (int): Number of roi copy into image.
        save_txt (bool): Save label.txt  True:Save/False:No save.
        save_img (bool): Save image.jpg  True:Save/False:No save.
        use_method (str): use method(opencv/mask/both) to copy roi into images.
        label_dict (dict): the label dictionary.
        img_path_list (list): the image path list.
    """
    def __init__(self,args):
        self.args = args
        #self.roi_label = self.args.roi_label
        self.img_dir = self.args.img_dir
        self.label_dir = self.args.label_dir
        self.roi_dir = self.args.roi_dir
        self.roi_mask_dir = self.args.roi_maskdir
        self.num_of_roi_into_image = self.args.num_roi
        self.save_txt = self.args.save_txt
        self.save_img = self.args.save_img
        self.label_dict = {"perdestrian":0,     "rider":1,      "car":2,    "train":3,      "t-light":4,    "stop-sign":5,      "lane marking":6}    
  
        self.drivable_area_label_dir =  self.args.dri_dir

        self.use_method = self.args.use_method

        self.img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))

    def ROI_Position_Criteria(self):

        #im = cv2.imread(img_path)
        h,w = self.im.shape[0],self.im.shape[1]
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        return x,y
     

    def Get_The_Position_Of_ROI_In_Images(self):
        x,y = self.ROI_Position_Criteria()
        return x,y
    

    def ROI_Size_Critiria(self):
        h,w = self.im.shape[0],self.im.shape[1]
        roi_w = random.randint(0,w-1)
        roi_h = random.randint(0,h-1)
        return roi_w,roi_h


    def Get_The_Size_Of_ROI_In_Images(self):
        roi_w,roi_h = self.ROI_Size_Critiria()
        return roi_w,roi_h
    
    def Update_YOLO_Label_txt(self):
        return NotImplemented


    def Update_Drivable_Map(self):
        return NotImplemented
    
    def Copy_roi_into_image(self):
        label = self.label_dict[self.roi_label]
        x,y=self.Get_The_Position_Of_ROI_In_Images()
        w,h=self.Get_The_Size_Of_ROI_In_Images
        label_info = (label,x,y,w,h)
        ret_label = self.Update_YOLO_Label_txt(label_info)
        ret_dri = self.Update_Drivable_Map(label_info)
        return NotImplemented
    
    


class Stop_Sign_Datasets(BaseDatasets):
    def Parsing_Img_list(self):
        for img_path in self.img_path_list:

            print(img_path)
            self.im = cv2.imread(img_path)
            stop_sign_x,stop_sign_y=self.Get_The_Position_Of_ROI_In_Images()
            print("stop_sign {},{}".format(stop_sign_x,stop_sign_y))
          


class Lane_Marking_Datasets(BaseDatasets):
    def Parsing_Img_list(self):
        for img_path in self.img_path_list:

            print(img_path)
            self.im = cv2.imread(img_path)
            lm_x,lm_y=self.Get_The_Position_Of_ROI_In_Images()
            print("lane_marking {},{}".format(lm_x,lm_y))


class Pedestrain_Datasets(BaseDatasets):
    def Parsing_Img_list(self):
        for img_path in self.img_path_list:

            print(img_path)
            self.im = cv2.imread(img_path)
            lm_x,lm_y=self.Get_The_Position_Of_ROI_In_Images()
            print("Pedestrain_marking {},{}".format(lm_x,lm_y))
          

#==============================================================================================================================================


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
    #==== For new class=========================================================================================================
    parser.add_argument('-roilabel','--roi-label',type=str,default="stop sign",help='the roi label')
    parser.add_argument('-roidir','--roi-dir',help='roi dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/roi")
    parser.add_argument('-roimaskdir','--roi-maskdir',help='roi mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/mask")
    parser.add_argument('-numroi','--num-roi',type=int,default=2,help='number of roi copy into image')
    #parser.add_argument('-numroi','--num-roi',type=int,default=2,help='number of roi copy into image')
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    parser.add_argument('-usemethod','--use-method',type=str,default="mask",help='opencv/mask/both')
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
    #FINISHED = Generate_LaneMarking_StopSign_Imgs(args_stopsign,
    #                                           args_roadsign)
    
    Stop_Sign_images = Stop_Sign_Datasets(args_stopsign)

    Stop_Sign_images.Parsing_Img_list()
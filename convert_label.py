import os
import glob
import shutil
import cv2

def Analysis_path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    file_name_rm_0 = file_name.split("0")[-1]
    file_rm_0 = file_name_rm_0 + ".jpg"
    file_dir = os.path.dirname(path)
    file_dir_dir = os.path.dirname(file_dir)
    file_dir_dir_name = os.path.basename(file_dir_dir)
    #print(file_dir_dir)
    #print(file_dir_dir_name)
    print("file_dir_dir:{}".format(file_dir_dir))
    print("file_dir_dir_name:{}".format(file_dir_dir_name))
    print("file:{}".format(file))
    print("file_name:{}".format(file_name))
    print("file_name_rm_0:{}".format(file_name_rm_0))
    print("file_rm_0:{}".format(file_rm_0))

    return file_dir_dir,file_dir_dir_name,file_name_rm_0,file_rm_0

def convert_MOT17_label(label_dir=None,
                        conver_format="yolo",
                        save_label_dir=None,
                        wanted_cls=None,
                        map_coco_cls=None):
    
    label_path_list = glob.glob(os.path.join(label_dir,"***/**/*.txt"))
    img_path_list = glob.glob(os.path.join(label_dir,"***/**/*.jpg"))
    #save images 
    for img_path in img_path_list:
        print(img_path)
        file_dir_dir,file_dir_dir_name,file_name_rm_0,file_rm_0 = Analysis_path(img_path)
        save_img_file_dir = os.path.join(save_label_dir,file_dir_dir_name,"images")
        os.makedirs(save_img_file_dir,exist_ok=True)
        new_img_path = os.path.join(save_img_file_dir,file_rm_0)
        shutil.copy(img_path,new_img_path)

    #parse labels to yolo format
    GONVERT_YOLO_TXT=True
    if GONVERT_YOLO_TXT:
        for label_path in label_path_list:
            print(label_path)
            file_dir_dir,file_dir_dir_name,_,_ = Analysis_path(label_path)
            #print(file_dir_dir)
            #print(file_dir_dir_name)
            #input()
            with open(label_path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    print(line) #get 13,1,912,484,97,109,0,7,1
                    parse_line = line.split(",")
                    frame = parse_line[0]
                    #get image info
                    #try:
                    img_file = frame + ".jpg"
                    img_file_dir = os.path.join(save_label_dir,file_dir_dir_name)
                    img_file_path = os.path.join(img_file_dir,"images",img_file)
                    
                    if os.path.exists(img_file_path):

                        img = cv2.imread(img_file_path)
                        img_w,img_h = img.shape[1],img.shape[0]
                    
                        c =  parse_line[-2]
                        
                        if int(c) in wanted_cls:
                            id = parse_line[1]
                            x_lt = parse_line[2] if int(parse_line[2])>0 else 0
                            y_lt = parse_line[3] if int(parse_line[3])>0 else 0
                            w = str( float(int(float(int(parse_line[4])/img_w)*1000000) / 1000000) )
                            h = str( float(int(float(int(parse_line[5])/img_h)*1000000) / 1000000) )
                            print("fr:{},id:{},x:{},y:{},w:{},h:{}".format(frame,id,x_lt,y_lt,w,h))
                            yolo_txt = frame + ".txt" #need add 000001.jpg
                            save_yolo_file_path = os.path.join(save_label_dir,file_dir_dir_name,"labels",yolo_txt)
                            save_yolo_file_dir = os.path.join(save_label_dir,file_dir_dir_name,"labels")
                            x  = str( float(  int(float((int(x_lt) + float(parse_line[4])/2.0)/img_w)*1000000) /1000000))
                            y  = str( float(  int(float((int(y_lt) + float(parse_line[5])/2.0)/img_h)*1000000) /1000000))
                            
                            
                            line = map_coco_cls[c] + " " + x + " " + y + " " + w + " " + h
                            
                            print(line)
                            os.makedirs(save_label_dir,exist_ok=True)
                            os.makedirs(save_yolo_file_dir,exist_ok=True)
                            with open(save_yolo_file_path,"a") as f_yolo:
                                f_yolo.write(line)
                                f_yolo.write("\n")
                            
                            f_yolo.close()
                    #except:
                        #pass


                

    #return NotImplementedError

def get_args():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("-labeldir","--label_dir",help="label directory",default="/home/ali/Projects/datasets/MOT/MOT17Det/train")
    return parser.parse_args()

if __name__=="__main__":
    #args = get_args()
    label_dir="/home/ali/Projects/datasets/MOT/MOT17Det/train"
    save_label_dir="/home/ali/Projects/datasets/MOT/MOT17DetLabels_ForYolo"
    wanted_cls = [1,2,3,4,5,7]
    map_coco_cls    = {"1":"0",     "2":"0",    "3":"2",    "4":"1",    "5":"3",    "7":"0"}
    is_failed = convert_MOT17_label(label_dir=label_dir,
                    conver_format="yolo",
                    save_label_dir=save_label_dir,
                    wanted_cls=wanted_cls,
                    map_coco_cls=map_coco_cls)
    #=======   MOTO labels ===========
    # Pedestrian 1
    # Person on vehicle 2
    # Car 3
    # Bicycle 4
    # Motorbike 5
    # Non motorized vehicle 6
    # Static person 7
    # Distractor 8
    # Occluder 9
    # Occluder on the ground 10
    # Occluder full 11
    # Reflection 12

    #=======coco lables==============
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
    #...
   
    
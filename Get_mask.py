import os
import glob
import cv2
import shutil

def parse_path(path):
    file = path.split("/")[-1]
    return file


def Get_mask(roi_dir=None,
             mask_dir=None):
    roi_path_list = glob.glob(os.path.join(roi_dir,"*.jpg"))
    c = 1

    os.makedirs(roi_dir+"/mask",exist_ok=True)
    save_dir = os.path.join(roi_dir,"mask")
    for roi_path in roi_path_list:
        print("{}:{}".format(c,roi_path))
       
        file = parse_path(roi_path)
        try:
            shutil.move(os.path.join(mask_dir,file),save_dir)
            print("{} : move mask successful".format(c))
            c+=1
        except:
            print("No mask found!!")
            c+=1
            pass

if __name__=="__main__":
    roi_dir = "./runs/predict/1_filter_by_ali/0/roi"
    mask_dir = "./runs/predict/1/mask"
    Get_mask(roi_dir=roi_dir,
             mask_dir=mask_dir)
import cv2
import glob
import shutil
import os

img_path_list = glob.glob(os.path.join("/home/ali/Pictures","*.png"))
c=5
for img_path in img_path_list:
    print(img_path)
    jpg_img = cv2.imread(img_path)
    save_dir = "/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/test_img"
    cv2.imwrite(save_dir+"/"+str(c)+".jpg",jpg_img)
    c+=1
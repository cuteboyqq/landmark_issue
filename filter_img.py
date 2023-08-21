import os
import shutil
import cv2
import glob
def Parse_Path(path):
    file = path.split("/")[-1]
    file_name = file.split(".")[0]
    return file,file_name

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
        if h*w < 60*60:
            print("too small (<60x60 pixels)")
        else:
            roi_dir = os.path.join(save_dir,"roi")
            shutil.copy(img_path,roi_dir)
            print("save img")
        
            if mask:
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                _,img_binary = cv2.threshold(img_gray,180,255,0)
                mask_dir = os.path.join(save_dir,"mask")
                cv2.imwrite(mask_dir+"/"+file,img_binary)
                print("{}:Save mask".format(c))
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
    save_dir = "./datasets/landmark_roi_filtered"
    FilterImg(img_dir,
              save_dir, 
              mask=True)
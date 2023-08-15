import cv2
import os

label_path = "/home/jnr_loganvo/Alister/GitHub_Code/landmark_issue/fake_landmark_image_test/labels/1ab901af-ab18a170.png"
label = cv2.imread(label_path)

# names_drive:
#     0: direct area
#     1: alternative area
#     2: background
#     3: landmark
print(label.shape)
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        print(label[i][j])
        if label[i][j][0]==3:
            input()
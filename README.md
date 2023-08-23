# landmark_issue

## [2023-08-19] add classify model to classify landmark and others images

## generate new image with landmark
```
python generate_arrow_img.py
```
### you can set parameter, the detail parameter is below
```
usage: generate_arrow_img.py [-h] [-imgdir IMG_DIR] [-drilabeldir DRI_LABELDIR] [-drilabeldirtrain DRI_LABELDIRTRAIN] [-linelabeldir LINE_LABELDIR]
                             [-roidir ROI_DIR] [-roimaskdir ROI_MASKDIR] [-saveimg] [-savecolormap] [-savetxt] [-numimg NUM_IMG] [-useopencvratio USE_OPENCVRATIO]
                             [-usemaskonly USE_MASK] [-show]

optional arguments:
  -h, --help            show this help message and exit
  -imgdir IMG_DIR, --img-dir IMG_DIR
                        image dir
  -drilabeldir DRI_LABELDIR, --dri-labeldir DRI_LABELDIR
                        drivable label dir
  -drilabeldirtrain DRI_LABELDIRTRAIN, --dri-labeldirtrain DRI_LABELDIRTRAIN
                        drivable label dir fo train
  -linelabeldir LINE_LABELDIR, --line-labeldir LINE_LABELDIR
                        line label dir
  -roidir ROI_DIR, --roi-dir ROI_DIR
                        roi dir
  -roimaskdir ROI_MASKDIR, --roi-maskdir ROI_MASKDIR
                        roi mask dir
  -saveimg, --save-img  save landmark fake images
  -savecolormap, --save-colormap
                        save generate semantic segment colormaps
  -savetxt, --save-txt  save landmark fake label.txt in yolo format cxywh
  -numimg NUM_IMG, --num-img NUM_IMG
                        number of generate fake landmark images
  -useopencvratio USE_OPENCVRATIO, --use_opencvratio USE_OPENCVRATIO
                        ratio of using opencv method to generate landmark images
  -usemaskonly USE_MASK, --use-mask USE_MASK
                        use mask method to generate landmark or not
  -show, --show         show images result

```
for example, using deault setting : 
```
def get_args():
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
    parser.add_argument('-savetxt','--save-txt',action='store_true',help='save landmark fake label.txt in yolo format cxywh')
    parser.add_argument('-numimg','--num-img',type=int,default=40000,help='number of generate fake landmark images')
    parser.add_argument('-useopencvratio','--use_opencvratio',type=float,default=0.50,help='ratio of using opencv method to generate landmark images')
    parser.add_argument('-usemaskonly','--use-mask',type=bool,default=True,help='use mask method to generate landmark or not')
    parser.add_argument('-show','--show',action='store_true',help='show images result')
   
    return parser.parse_args()  

```


## generate landmark ROI
### Using bdd100k dataset images, integrate landmark ROI into bdd100k images
```
python img_process.py
```
### bdd100k dataset directory format
```
.
├── images
│   ├── 100k
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── 10k
│       ├── test
│       ├── train
│       └── val
├── labels
│   ├── det_20
│   │   ├── bdd100k_format
│   │   │   ├── det_train.json
│   │   │   └── det_val.json
│   │   ├── train.json
│   │   └── val.json
│   ├── detection
│   │   ├── train
│   │   └── val
│   ├── drivable
│   │   ├── colormaps
│   │   │   ├── train
│   │   │   └── val
│   │   ├── masks
│   │   │   ├── train
│   │   │   └── val
│   │   ├── polygons
│   │   │   ├── drivable_train.json
│   │   │   └── drivable_val.json
│   │   └── rles
│   │       ├── drivable_train.json
│   │       └── drivable_val.json
│   ├── drivable_lane
│   │   ├── color_masks
│   │   │   ├── train
│   │   │   └── val
│   │   └── masks
│   │       ├── train
│   │       └── val
│   ├── ins_seg
│   │   ├── bitmasks
│   │   │   ├── train
│   │   │   └── val
│   │   ├── colormaps
│   │   │   ├── train
│   │   │   └── val
│   │   ├── polygons
│   │   │   ├── ins_seg_train.json
│   │   │   └── ins_seg_val.json
│   │   └── rles
│   │       ├── ins_seg_train.json
│   │       └── ins_seg_val.json
│   └── lane
│       ├── colormaps
│       │   ├── train
│       │   └── val
│       ├── masks
│       │   ├── train
│       │   └── val
│       └── polygons
│           ├── lane_train.json
│           └── lane_val.json
├── train.txt
└── val.txt
```


## result images
![0ff22e9f-bc99e801](https://github.com/cuteboyqq/landmark_issue/assets/58428559/58bae0d4-6044-4b6c-a2c7-f15798b05319)


![01a4933a-252e566a](https://github.com/cuteboyqq/landmark_issue/assets/58428559/fe18bc06-db9a-4c37-a30e-941cacab9839)


![0c4d4afa-75fce0c1](https://github.com/cuteboyqq/landmark_issue/assets/58428559/0f571ef5-e9c9-4c64-aeab-dd373f6ff3c0)


![1f79f230-a15dcf16](https://github.com/cuteboyqq/landmark_issue/assets/58428559/f9ceaadf-ae78-4658-9308-a2feb5e2fe68)
![1f93fff6-8be58b84](https://github.com/cuteboyqq/landmark_issue/assets/58428559/f39425db-39bb-464a-bade-5d8d5f6f919b)
![2dc745eb-8c897bb9](https://github.com/cuteboyqq/landmark_issue/assets/58428559/caf2ba36-3422-45c0-a60c-c292f4ec5cfb)
![2d86b379-c3a6e578](https://github.com/cuteboyqq/landmark_issue/assets/58428559/10c8f1b9-dd69-4106-aab6-0f209238b5e1)
![02de1e9e-77fd472f](https://github.com/cuteboyqq/landmark_issue/assets/58428559/12196eff-806c-4e92-a006-ea06bee3e815)
![3a686dcb-bc55c6d1](https://github.com/cuteboyqq/landmark_issue/assets/58428559/5e911c9e-724e-4bc3-a81b-ce079c368e6d)
![3cea8608-b550c2df](https://github.com/cuteboyqq/landmark_issue/assets/58428559/ba178e9c-a1c0-46e3-aed6-42bcb41a0838)
![3fdd18b1-451986b4](https://github.com/cuteboyqq/landmark_issue/assets/58428559/75e2ca8e-f50b-4126-b5a1-66ee70212c6a)
![6ab99703-03398472](https://github.com/cuteboyqq/landmark_issue/assets/58428559/411ddbcf-76d1-45fa-992a-0c43dc0c1d5c)
![6f950c73-219dd833](https://github.com/cuteboyqq/landmark_issue/assets/58428559/9e9a6e51-d453-41f0-96b6-38220ed75568)
![7c618810-ea173340](https://github.com/cuteboyqq/landmark_issue/assets/58428559/0e21f984-a059-4ed4-85dd-690f4b18adcc)
![9275c992-6ee66800](https://github.com/cuteboyqq/landmark_issue/assets/58428559/97c990ba-986e-4455-ad4e-d938adc1261a)
![52846a90-0a740715](https://github.com/cuteboyqq/landmark_issue/assets/58428559/caa4c2e9-fe7a-47df-af33-e25395a03969)
![44908b6e-2df016c4](https://github.com/cuteboyqq/landmark_issue/assets/58428559/6befa6db-2620-4585-9c05-fbdaede4814a)

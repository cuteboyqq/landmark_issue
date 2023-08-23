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



## reslut images
![0a7b0436-1a46bb12](https://github.com/cuteboyqq/landmark_issue/assets/58428559/53e3b84f-99d0-4d1d-97ce-6c5361e1836a)
![0a14abcb-e1a9d5d9](https://github.com/cuteboyqq/landmark_issue/assets/58428559/8ac83daf-34c6-40d3-8e8c-1e3fc7f0f059)
![0aa842d5-a319d2e2](https://github.com/cuteboyqq/landmark_issue/assets/58428559/78a8cf19-a172-4039-ade7-35b49e5d9a31)
![0aec42a1-17a32b5a](https://github.com/cuteboyqq/landmark_issue/assets/58428559/a8cf5c3b-a73a-48eb-97ef-091ecf4cc21f)
![0b24b165-efec69bb](https://github.com/cuteboyqq/landmark_issue/assets/58428559/2547690a-b80e-4718-bc39-cad6eade791f)
![0b71f4fc-d83ed49c](https://github.com/cuteboyqq/landmark_issue/assets/58428559/4a00fd20-d2f6-4d29-b20f-fc40523b0627)


![0b6781f6-eeb5fc5f](https://github.com/cuteboyqq/landmark_issue/assets/58428559/e4b4366c-f4e6-41be-ae94-63d66c7cc71b)
![0bdf6421-0cca8ba2](https://github.com/cuteboyqq/landmark_issue/assets/58428559/08e9eda9-9d2f-4dda-920c-93beb27e504f)
![0bf28365-020515e2](https://github.com/cuteboyqq/landmark_issue/assets/58428559/4d894ebc-8ec9-401b-9c23-84f85d1bc2a8)
![0c597dc1-bc5a9586](https://github.com/cuteboyqq/landmark_issue/assets/58428559/ab962776-5ee1-4783-9983-c6ccf9902392)
![0ca7d4fe-7eaf5269](https://github.com/cuteboyqq/landmark_issue/assets/58428559/ce8402c8-002a-4b6b-aa68-79642598efd7)

![004071a4-4e8a363a](https://github.com/cuteboyqq/landmark_issue/assets/58428559/222c3b86-2058-40a6-8ab6-cce471c49559)
![4075c629-361acafb](https://github.com/cuteboyqq/landmark_issue/assets/58428559/5360daab-327f-4d7f-b4a4-53ee11742791)
![4080d966-aceadb52](https://github.com/cuteboyqq/landmark_issue/assets/58428559/843f2ccb-1ea9-49f0-b7f7-fbe5ba90fc12)
![4097ad84-74b4838c](https://github.com/cuteboyqq/landmark_issue/assets/58428559/bb13d348-6a60-413f-97ee-93de8b78fbf5)
![4170f5e5-9b11385b](https://github.com/cuteboyqq/landmark_issue/assets/58428559/a33afcb9-4ef3-4d8c-94a2-bf36c452ad42)
![4178c48e-b9faf25c](https://github.com/cuteboyqq/landmark_issue/assets/58428559/dba901be-ddc9-4a34-9a57-010fbd80f07a)
![4179b0c0-15e91828](https://github.com/cuteboyqq/landmark_issue/assets/58428559/e3e4477f-1eb3-4c91-80c5-a1aaa12217f2)
![04200e90-3ca8cd17](https://github.com/cuteboyqq/landmark_issue/assets/58428559/cd551e83-9d6f-453b-9570-0890dfcaa9ee)

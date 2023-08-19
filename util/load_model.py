# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:06:04 2022

@author: User
"""
from models.resnet import ResNet,ResBlock
from models.repVGG import RepVGG,RepVGGBlock
from models.res2net import Res2Net,Bottle2neck
from models.VGG16 import VGG16
from models.shufflenet import *
from models.efficientnet import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.lenet import *
from models.densenet import *
from models.shufflenetv2 import *

def load_model(opts,nc):
    if opts.model=='resnet' or opts.model=='Resnet' or opts.model=='ResNet':
        model = ResNet(ResBlock,nc=nc)
    elif opts.model=='repvgg' or opts.model=='RepVGG' or opts.model=='Repvgg' or opts.model=='RepVgg' or opts.model=='repVgg' or opts.model=='repVGG':
        model = RepVGG(num_classes=10)
    elif opts.model=='vgg16' or opts.model=='VGG16' or opts.model=='Vgg16':
        model = VGG16()
    elif opts.model=='res2net' or opts.model=='Res2net' or opts.model=='Res2Net':
        model = Res2Net() 
    elif opts.model=='shufflenet' or opts.model=='ShuffleNet' or opts.model=='shuffleNet':
        model = ShuffleNetG2()
    elif opts.model=='EfficientNet' or opts.model=='efficientNet' or opts.model=='efficientnet':
        model = EfficientNetB0()
    elif opts.model=='MobileNet' or opts.model=='Mobilenet' or opts.model=='mobilenet':
        model = MobileNet()
    elif opts.model=='MobileNetV2' or opts.model=='MobileNetv2' or opts.model=='mobileNetv2' or opts.model=='mobilenetv2':
        model = MobileNetV2()
    elif opts.model=='LeNet' or opts.model=='Lenet' or opts.model=='lenet' or opts.model=='leNet':
        model = LeNet()
    elif opts.model=='DenseNet' or opts.model=='Densenet' or opts.model=='denseNet' or opts.model=='densenet':
        model = densenet_cifar()
    elif opts.model=='ShuffleNetV2' or opts.model=='shuffleNetV2' or opts.model=='shufflenetV2' or opts.model=='shufflenetv2':
        model = ShuffleNetV2(net_size=0.5)
        
    return model
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 01:49:35 2022

@author: User
"""
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torchvision


def load_data(opts):
    batch_size = opts.batch_size
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    
    if opts.data=='mnist' or opts.data=='mnist_data' :
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                                                  transform=transforms.ToTensor()),
                                                   batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True,
                                                                  transform=transforms.ToTensor()),
                                                   batch_size=batch_size, shuffle=False, **kwargs)
        
    elif opts.data=='cifar10' or opts.data=='cifar10_data':
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='../data', train=True, download=True, 
                                                                   transform=transform_train),
                                                  batch_size=batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test),
                                                 batch_size=batch_size, shuffle=False, num_workers=1)
    else: #custom datasets
        size = (opts.img_size,opts.img_size)
        train_data = torchvision.datasets.ImageFolder(opts.data,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(size),
                                                        transforms.RandomHorizontalFlip(),
                                                        #transforms.Scale(64),
                                                        transforms.CenterCrop(size),
                                                     
                                                        transforms.ToTensor()
                                                        ])
                                                    )
        test_data = torchvision.datasets.ImageFolder(opts.data_test,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(size),
                                                        transforms.RandomHorizontalFlip(),
                                                        #transforms.Scale(64),
                                                        transforms.CenterCrop(size),
                                                     
                                                        transforms.ToTensor()
                                                        ])
                                                    )
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True,drop_last=False)
        test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader
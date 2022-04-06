#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:46:33 2017

@author: tiaotiao
"""

from PIL import Image
import os
import os.path
import random
import cv2

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class ClassificationImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, classification_file_name, dataset_source_path, transform=True, loader=default_image_loader):

        self.base_path = base_path  
        classification_file_address = base_path + classification_file_name
        
        self.dataset_source_path = dataset_source_path

        classifications = []
        for line in open(classification_file_address):
            add = line.split(',')[0]
            label = int(line.split(',')[1])-1
            classifications.append((add, label))
        self.classifications = classifications
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.classifications[index]
        path = self.dataset_source_path + path
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.classifications)

class ClassificationImageLoaderTriplet(torch.utils.data.Dataset):
    def __init__(self, base_path, classification_file_name, dataset_source_path, transform=True, loader=default_image_loader):

        self.base_path = base_path  
        classification_file_address = base_path + classification_file_name
        
        self.dataset_source_path = dataset_source_path

        Add = []
        Label = []

        for line in open(classification_file_address):
            add = line.split(',')[0]
            label = int(line.split(',')[1])-1
            Add.append(add)
            Label.append(label)
        
        Data = dict(zip(Add, Label))

        self.Add = Add
        self.Label = Label
        self.Data = Data

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        #get target
        path_t = self.Add[index]
        label_t = self.Label[index]
        path_t = self.dataset_source_path + path_t
        img_t = self.loader(path_t)

        #get anchor
        data_anchor = {k:v for k,v in self.Data.items()  if v==label_t}
        anchor = random.sample(data_anchor.keys(), 1)
        path_a = anchor[0]
        label_a = data_anchor[path_a]
        path_a = self.dataset_source_path + path_a
        img_a = self.loader(path_a)

        #get negative
        data_negative = {k:v for k,v in self.Data.items()  if v!=label_t}
        negative = random.sample(data_negative.keys(), 1)
        path_n = negative[0]
        label_n = data_negative[path_n]
        path_n = self.dataset_source_path + path_n
        img_n = self.loader(path_n)

        if self.transform is not None:
            img_t = self.transform(img_t)
            img_a = self.transform(img_a)
            img_n = self.transform(img_n)

        return img_t, img_a, img_n, label_t, label_a, label_n 

    def __len__(self):
        return len(self.Add)
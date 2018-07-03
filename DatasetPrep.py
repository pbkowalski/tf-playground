# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:48:43 2018

@author: pk0300
"""

import pandas as pd
import numpy as np
from os import listdir # get list of all the files in a directory
from os.path import isfile, join # to manupulate file paths
from random import shuffle # shuffle the data (file paths)
import cv2 # to load the images

data_dir = 'img_sorted'

class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths
    
    def get_mini_batches(self, batch_size=10, image_size=(85, 68), allchannel=True):
        images = []
        labels = []
        empty=False
        counter=0
        each_batch_size=int(batch_size/len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) < counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (counter)%each_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]
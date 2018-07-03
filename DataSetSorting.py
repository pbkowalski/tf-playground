# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:48:43 2018

@author: pk0300
"""

import pandas as pd
import numpy as np
from os import listdir # get list of all the files in a directory
from os.path import isfile, join # to manupulate file paths
from shutil import copyfile

data_dir = 'img'
output_dir = 'img_sorted'
output_tr = 'img_tr'

output_tst = 'img_tst'

df = pd.read_csv('AllBirdsv4.csv')
cols = df.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
df.columns = cols
nameMap = dict(zip(list(df.File_ID), list(df.English_name)))
labels = np.unique(list(nameMap.values()))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_tr):
    os.makedirs(output_tr)
if not os.path.exists(output_tst):
    os.makedirs(output_tst)
    
for filename in listdir(data_dir):
    if isfile(join(data_dir, filename)):
        tokens = filename.split('.')
        if tokens[-1] == 'jpg':
            img_path = join(data_dir, filename)
            label = nameMap[int(tokens[0])]
            if (random.random()<0.7):
                out = output_tr
            else:
                out = output_tst
                
            if not os.path.exists(join(out, label)):
                os.makedirs(join(out, label))
            target = join(join(out, label), filename)
            if not os.path.exists(target):
                copyfile(img_path, target)
                
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:58:20 2020

@author: chenlin
"""

import torch
import cv2
import numpy as np

ps_datapath = r"F:\TrainingData\PSDataset\4\patchImg.bin"
nn = torch.load(ps_datapath)

patch_example = nn.cpu().numpy()[7,0,:,:]
cv2.imwrite('F:/TrainingData/PSDataset/4/patch_example7.jpg', patch_example)

ps_datapath = r"F:\TrainingData\PSDataset\4\patch_info.txt"
patch_info= np.loadtxt(ps_datapath,delimiter=',')


brown_datapath = r"F:\hardnet\hardnet_ipi\data\sets\yosemite.pt"
nnb = torch.load(brown_datapath)

#how to convert patch info 3D index and other information to the form that 
# hardnet dataset also compatitable with
# put the data, label and matches in pt file 

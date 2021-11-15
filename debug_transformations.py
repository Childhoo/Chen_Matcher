# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:49:43 2019

@author: chenlin
"""
import torch
import numpy as np
#import h5py
import math
from augmentation import get_random_norm_affine_LAFs,get_random_rotation_LAFs, get_random_shifts_LAFs

#read data first
d = torch.load("F:/affnet/affnet/dataset/ipi_data/ipi_dortmund5.pt")
data = d[0]
#patches = np.asarray(data["x"])
#data = patches

rot_LAFs, inv_rotmat = get_random_rotation_LAFs(data, math.pi)


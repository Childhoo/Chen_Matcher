# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:44:47 2019

@author: chenlin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:12:58 2019

@author: chenlin
"""


import os
import errno
import numpy as np
from PIL import Image
import torchvision.datasets as dset

import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import str2bool

#from dataset import  TripletPhotoTour
from dataset_ipi_graz import TripletPhotoTour_IPI_Graz
root='/home/chen/data/TrainingData'
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour_IPI_Graz(train=True,
                         batch_size=128,
                         root=root,
                         name='ipi_graz',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)


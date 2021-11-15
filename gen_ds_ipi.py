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
from dataset_ipi import TripletPhotoTour_IPI
root='dataset/ipi_data'
train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour_IPI(train=True,
                         batch_size=128,
                         root=root,
                         name='ipi_dortmund5',
                         download=True,
                         transform=None),
                         batch_size=128,
                         shuffle=False)

